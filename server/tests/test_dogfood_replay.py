from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from engram.evaluation.dogfood import (
    DOGFOOD_REPORT_KIND,
    add_dogfood_session_label,
    build_dogfood_candidate_report,
    build_dogfood_closeout_report,
    build_dogfood_human_label_evidence_artifact,
    build_dogfood_label_import_samples,
    build_dogfood_label_template,
    build_dogfood_replay_report,
    build_dogfood_review_report,
    build_dogfood_turn_inspection_report,
    dogfood_memory_operation_metrics_from_replay_report,
    export_dogfood_human_label_evidence,
    finalize_dogfood_labels,
    import_dogfood_label_artifact,
    import_dogfood_replay_cost_metrics,
    parse_modes,
    parse_transcript_text,
    prepare_dogfood_review_bundle,
    render_dogfood_candidate_markdown,
    render_dogfood_closeout_markdown,
    render_dogfood_export_markdown,
    render_dogfood_finalize_markdown,
    render_dogfood_label_edit_markdown,
    render_dogfood_prepare_markdown,
    render_dogfood_replay_markdown,
    render_dogfood_review_markdown,
    render_dogfood_turn_inspection_markdown,
    run_dogfood_command,
    update_dogfood_turn_label,
)
from engram.evaluation.human_label_evidence import build_human_label_evidence
from engram.evaluation.store import SQLiteEvaluationStore


def test_parse_transcript_text_accepts_simple_jsonl_and_axi_traces() -> None:
    raw = "\n".join(
        [
            json.dumps({"role": "user", "content": "What did we decide on Engram?"}),
            json.dumps({"role": "assistant", "content": "We chose AXI."}),
            json.dumps({"operation": "context", "status": "ok", "durationMs": 12}),
        ]
    )

    turns = parse_transcript_text(raw, source="fixture.jsonl")

    assert [turn.role for turn in turns] == ["user", "assistant", "tool"]
    assert turns[0].content == "What did we decide on Engram?"
    assert turns[2].content == "context ok"


def test_parse_transcript_text_accepts_codex_response_item_jsonl() -> None:
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Can Engram replay real Codex transcripts?",
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Reply exactly OK.",
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Yes, with redaction by default.",
                            }
                        ],
                    },
                }
            ),
        ]
    )

    turns = parse_transcript_text(raw, source="codex.jsonl")

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "Can Engram replay real Codex transcripts?"
    assert turns[1].content == "Yes, with redaction by default."


def test_parse_transcript_text_skips_codex_bootstrap_user_messages() -> None:
    raw = "\n".join(
        [
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "# AGENTS.md instructions for "
                                    "/Users/konnermoshier/Engram\n\n<INSTRUCTIONS>"
                                ),
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "<goal_context>\n"
                                    "Continue working toward the active thread goal."
                                ),
                            }
                        ],
                    },
                }
            ),
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": ("Can Engram prepare dogfood labels from this transcript?"),
                            }
                        ],
                    },
                }
            ),
        ]
    )

    turns = parse_transcript_text(raw, source="codex.jsonl")

    assert [turn.content for turn in turns] == [
        "Can Engram prepare dogfood labels from this transcript?"
    ]


def test_parse_transcript_text_accepts_markdown_roles() -> None:
    turns = parse_transcript_text(
        """
        User: Catch me up on the native Helix work.
        Assistant: The PyO3 path is primary.
        """,
        source="fixture.md",
    )

    assert len(turns) == 2
    assert turns[0].role == "user"
    assert "native Helix" in turns[0].content


def test_parse_modes_rejects_unknown_modes() -> None:
    with pytest.raises(SystemExit):
        parse_modes("off,magic")


@pytest.mark.asyncio
async def test_build_dogfood_replay_report_redacts_content_by_default(tmp_path) -> None:
    transcript = tmp_path / "turns.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Thanks"}),
                json.dumps({"role": "user", "content": "What changed with Engram AXI?"}),
            ]
        ),
        encoding="utf-8",
    )

    report = await build_dogfood_replay_report(
        transcript_path=transcript,
        project_path="/tmp/engram",
        group_id="native_brain",
        modes=["off", "gated_lite", "deep"],
    )

    assert report["kind"] == DOGFOOD_REPORT_KIND
    assert report["status"] == "measured"
    assert report["source"]["content_redacted"] is True
    assert report["source"]["user_turn_count"] == 2
    assert "content" not in report["turns"][0]
    assert report["turns"][1]["need"]["query_hint"] is None
    assert report["turns"][1]["need"]["query_hint_redacted"] is True
    assert report["turns"][0]["decisions"][1]["decision"] == "skipped"
    assert report["turns"][1]["decisions"][1]["decision"] == "triggered"
    assert report["mode_summaries"]["gated_lite"]["turn_count"] == 2
    assert report["labels"]["opt_in_required"] is True

    markdown = render_dogfood_replay_markdown(report)
    assert "Engram Dogfood Replay" in markdown
    assert "What changed" not in markdown


@pytest.mark.asyncio
async def test_build_dogfood_replay_report_summarizes_axi_trace_evidence(tmp_path) -> None:
    transcript = tmp_path / "axi-trace.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "What is Engram doing now?"}),
                json.dumps(
                    {
                        "operation": "home",
                        "status": "healthy",
                        "durationMs": 42,
                        "origin": "session-start-hook",
                        "client": "codex",
                        "timeoutSeconds": 3,
                        "cacheHit": False,
                    }
                ),
                json.dumps(
                    {
                        "operation": "context",
                        "status": "ok",
                        "durationMs": 80,
                        "origin": "agent-followup",
                        "client": "codex",
                        "timeoutSeconds": 5,
                        "cacheHit": True,
                        "packetCount": 3,
                        "resultCount": 0,
                        "fallbackStatus": "cache_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "operation": "recall",
                        "status": "degraded",
                        "durationMs": 5000,
                        "origin": "agent-followup",
                        "client": "codex",
                        "timeoutSeconds": 5,
                        "packetCount": 1,
                        "resultCount": 2,
                        "fallbackStatus": "recall_timeout",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = await build_dogfood_replay_report(
        transcript_path=transcript,
        project_path="/tmp/engram",
        group_id="native_brain",
        modes=["cached", "deep"],
    )

    trace = report["trace_evidence"]
    assert trace["status"] == "measured"
    assert trace["trace_count"] == 3
    assert trace["operation_counts"] == {"context": 1, "home": 1, "recall": 1}
    assert trace["status_counts"] == {"degraded": 1, "healthy": 1, "ok": 1}
    assert trace["origin_counts"] == {"agent-followup": 2, "session-start-hook": 1}
    assert trace["client_counts"] == {"codex": 3}
    assert trace["duration_ms"]["avg"] == 1707.3333
    assert trace["duration_ms"]["p95"] == 5000
    assert trace["timeout_count"] == 1
    assert trace["degraded_count"] == 1
    assert trace["cache_hit_count"] == 1
    assert trace["packet_count"] == 4
    assert trace["result_count"] == 2
    assert trace["fallback_status_counts"] == {
        "cache_satisfied": 1,
        "recall_timeout": 1,
    }
    assert trace["session_start_count"] == 1
    assert trace["followup_count"] == 2

    markdown = render_dogfood_replay_markdown(report)
    assert "## Trace Evidence" in markdown
    assert "AXI trace records: 3" in markdown
    assert "context=1" in markdown
    assert "Packets/results: 4/2" in markdown
    assert "cache_satisfied=1" in markdown
    assert "Degraded/timeouts: 1/1" in markdown


@pytest.mark.asyncio
async def test_build_dogfood_replay_report_merges_separate_axi_trace_file(
    tmp_path,
) -> None:
    transcript = tmp_path / "turns.jsonl"
    trace_path = tmp_path / "axi-trace.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "content": "What changed with Engram AXI?"}),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "operation": "home",
                        "status": "healthy",
                        "durationMs": 20,
                        "origin": "session-start-hook",
                        "client": "codex",
                    }
                ),
                json.dumps(
                    {
                        "operation": "context",
                        "status": "ok",
                        "durationMs": 60,
                        "origin": "agent-followup",
                        "client": "codex",
                        "cacheHit": True,
                        "packetCount": 2,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = await build_dogfood_replay_report(
        transcript_path=transcript,
        trace_paths=[trace_path],
        project_path="/tmp/engram",
        group_id="native_brain",
        modes=["cached"],
    )

    assert report["source"]["user_turn_count"] == 1
    assert report["source"]["turn_count"] == 3
    assert report["source"]["trace_paths"] == [str(trace_path)]
    assert report["trace_evidence"]["status"] == "measured"
    assert report["trace_evidence"]["trace_count"] == 2
    assert report["trace_evidence"]["operation_counts"] == {"context": 1, "home": 1}
    assert report["trace_evidence"]["cache_hit_count"] == 1
    assert report["trace_evidence"]["packet_count"] == 2


@pytest.mark.asyncio
async def test_build_dogfood_replay_report_filters_trace_evidence_by_since_and_project(
    tmp_path,
) -> None:
    transcript = tmp_path / "turns.jsonl"
    trace_path = tmp_path / "axi-trace.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "content": "What changed with Engram AXI?"}),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-26T18:00:00Z",
                        "operation": "home",
                        "status": "healthy",
                        "durationMs": 500,
                        "origin": "session-start-hook",
                        "project": "/tmp/engram",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-26T19:05:00Z",
                        "operation": "context",
                        "status": "ok",
                        "durationMs": 90,
                        "origin": "agent-followup",
                        "project": "/tmp/other",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-26T19:10:00Z",
                        "operation": "recall",
                        "status": "ok",
                        "durationMs": 70,
                        "origin": "agent-followup",
                        "project": "/tmp/engram/subdir",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = await build_dogfood_replay_report(
        transcript_path=transcript,
        trace_paths=[trace_path],
        project_path="/tmp/engram",
        group_id="native_brain",
        modes=["cached"],
        trace_since="2026-05-26T19:00:00Z",
        trace_project_path="/tmp/engram",
    )

    trace = report["trace_evidence"]
    assert trace["status"] == "measured"
    assert trace["raw_trace_count"] == 3
    assert trace["trace_count"] == 1
    assert trace["operation_counts"] == {"recall": 1}
    assert trace["duration_ms"]["avg"] == 70.0
    assert trace["filter"]["kept_trace_count"] == 1
    assert trace["filter"]["dropped_before_since"] == 1
    assert trace["filter"]["dropped_project_mismatch"] == 1

    markdown = render_dogfood_replay_markdown(report)
    assert "Trace filter: raw 3, kept 1" in markdown


@pytest.mark.asyncio
async def test_build_dogfood_replay_report_counts_trace_only_files_as_measured(
    tmp_path,
) -> None:
    transcript = tmp_path / "axi-trace-only.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "operation": "home",
                "status": "healthy",
                "durationMs": 20,
                "origin": "session-start-hook",
                "client": "codex",
            }
        ),
        encoding="utf-8",
    )

    report = await build_dogfood_replay_report(
        transcript_path=transcript,
        project_path="/tmp/engram",
        group_id="native_brain",
        modes=["cached"],
    )

    assert report["status"] == "measured"
    assert report["source"]["user_turn_count"] == 0
    assert report["trace_evidence"]["status"] == "measured"
    assert report["trace_evidence"]["trace_count"] == 1


@pytest.mark.asyncio
async def test_prepare_dogfood_review_bundle_writes_replay_labels_and_review(
    tmp_path,
) -> None:
    transcript = tmp_path / "turns.jsonl"
    trace_path = tmp_path / "axi-trace.jsonl"
    out_dir = tmp_path / "bundle"
    transcript.write_text(
        json.dumps({"role": "user", "content": "What changed with Engram AXI?"}),
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps({"operation": "context", "status": "ok", "durationMs": 42}),
        encoding="utf-8",
    )

    report = await prepare_dogfood_review_bundle(
        transcript_path=transcript,
        trace_paths=[trace_path],
        project_path="/tmp/engram",
        output_dir=out_dir,
        group_id="native_brain",
        modes=["cached", "deep"],
        include_template_content=True,
    )
    markdown = render_dogfood_prepare_markdown(report)
    paths = report["paths"]
    replay = json.loads(Path(paths["replay_report"]).read_text(encoding="utf-8"))
    labels = json.loads(Path(paths["labels"]).read_text(encoding="utf-8"))
    review = json.loads(Path(paths["review_report"]).read_text(encoding="utf-8"))

    assert report["status"] == "prepared"
    assert replay["source"]["content_redacted"] is True
    assert replay["turns"][0]["need"]["query_hint"] is None
    assert replay["turns"][0]["need"]["query_hint_redacted"] is True
    assert replay["trace_evidence"]["status"] == "measured"
    assert labels["source"]["content_redacted"] is False
    assert labels["turns"][0]["content"] == "What changed with Engram AXI?"
    assert labels["turns"][0]["query_hint"]
    assert labels["turns"][0]["query_hint_redacted"] is False
    assert review["status"] == "needs_labels"
    assert Path(paths["review_markdown"]).exists()
    assert "--replay-report" in report["next_commands"]["finalize"]
    assert "# Engram Dogfood Prepare" in markdown


@pytest.mark.asyncio
async def test_prepare_dogfood_review_bundle_preserves_redacted_query_hint_flags(
    tmp_path,
) -> None:
    transcript = tmp_path / "turns.jsonl"
    out_dir = tmp_path / "bundle"
    transcript.write_text(
        json.dumps({"role": "user", "content": "What changed with Engram AXI?"}),
        encoding="utf-8",
    )

    report = await prepare_dogfood_review_bundle(
        transcript_path=transcript,
        trace_paths=[],
        project_path="/tmp/engram",
        output_dir=out_dir,
        group_id="native_brain",
        modes=["cached", "deep"],
        include_template_content=False,
    )
    paths = report["paths"]
    replay = json.loads(Path(paths["replay_report"]).read_text(encoding="utf-8"))
    labels = json.loads(Path(paths["labels"]).read_text(encoding="utf-8"))
    review = json.loads(Path(paths["review_report"]).read_text(encoding="utf-8"))

    assert replay["turns"][0]["need"]["query_hint"] is None
    assert replay["turns"][0]["need"]["query_hint_redacted"] is True
    assert labels["turns"][0]["query_hint"] is None
    assert labels["turns"][0]["query_hint_redacted"] is True
    assert review["review"]["review_queue"][0]["query_hint"] is None
    assert review["review"]["review_queue"][0]["query_hint_redacted"] is True


@pytest.mark.asyncio
async def test_prepare_dogfood_review_bundle_marks_trace_only_without_labelable_turns(
    tmp_path,
) -> None:
    transcript = tmp_path / "turns.jsonl"
    trace_path = tmp_path / "axi-trace.jsonl"
    out_dir = tmp_path / "bundle"
    transcript.write_text(
        json.dumps({"role": "user", "content": "Reply exactly OK."}),
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps({"operation": "home", "status": "healthy", "durationMs": 42}),
        encoding="utf-8",
    )

    report = await prepare_dogfood_review_bundle(
        transcript_path=transcript,
        trace_paths=[trace_path],
        project_path="/tmp/engram",
        output_dir=out_dir,
        group_id="native_brain",
        modes=["cached"],
    )
    markdown = render_dogfood_prepare_markdown(report)

    assert report["status"] == "trace_only"
    assert report["replay"]["trace_status"] == "measured"
    assert report["labels"]["turn_count"] == 0
    assert report["next_commands"]["finalize"] is None
    assert report["review"]["status"] == "trace_only"
    assert "trace/cost evidence only" in markdown


def test_build_dogfood_candidate_report_finds_labelable_transcripts(tmp_path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    trace_only = tmp_path / "trace-only.jsonl"
    empty = tmp_path / "empty.jsonl"
    candidate.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "sess_candidate", "cwd": "/tmp/engram"},
                    }
                ),
                json.dumps({"role": "user", "content": "What did Engram remember?"}),
                json.dumps({"role": "assistant", "content": "It remembered the plan."}),
            ]
        ),
        encoding="utf-8",
    )
    trace_only.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Reply exactly OK."}),
                json.dumps({"operation": "home", "status": "healthy", "durationMs": 10}),
            ]
        ),
        encoding="utf-8",
    )
    empty.write_text(
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "<environment_context>\n  <cwd>/tmp</cwd>",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_candidate_report(
        root_path=tmp_path,
        project_path="/tmp/engram",
        limit=5,
        max_files=10,
    )
    markdown = render_dogfood_candidate_markdown(report)

    assert report["candidate_count"] == 1
    assert report["trace_only_count"] == 1
    assert report["skipped_count"] == 1
    assert report["candidates"][0]["path"] == str(candidate)
    assert report["candidates"][0]["labelable_turn_count"] == 1
    assert report["candidates"][0]["session_cwd"] == "/tmp/engram"
    assert report["candidates"][0]["project_match"] is True
    assert "What did Engram remember?" not in markdown
    assert "engram dogfood prepare" in markdown


def test_build_dogfood_candidate_report_can_filter_project_mismatches(tmp_path) -> None:
    match = tmp_path / "match.jsonl"
    mismatch = tmp_path / "mismatch.jsonl"
    match.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "sess_match", "cwd": "/tmp/engram/server"},
                    }
                ),
                json.dumps({"role": "user", "content": "Engram project question"}),
            ]
        ),
        encoding="utf-8",
    )
    mismatch.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "sess_other", "cwd": "/tmp/other"},
                    }
                ),
                json.dumps({"role": "user", "content": "Other project question"}),
            ]
        ),
        encoding="utf-8",
    )

    report = build_dogfood_candidate_report(
        root_path=tmp_path,
        project_path="/tmp/engram",
        project_only=True,
        limit=5,
        max_files=10,
    )

    assert report["candidate_count"] == 1
    assert report["project_mismatch_count"] == 1
    assert report["candidates"][0]["path"] == str(match)


def test_build_dogfood_candidate_report_uses_tool_workdir_for_project_match(
    tmp_path,
) -> None:
    transcript = tmp_path / "codex-home-session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "sess_home", "cwd": "/Users/konnermoshier"},
                    }
                ),
                json.dumps({"role": "user", "content": "Continue Engram dogfood"}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "function_call",
                            "name": "exec_command",
                            "arguments": json.dumps(
                                {
                                    "cmd": "pytest -q server/tests/test_dogfood_replay.py",
                                    "workdir": "/Users/konnermoshier/Engram",
                                }
                            ),
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = build_dogfood_candidate_report(
        root_path=tmp_path,
        project_path="/Users/konnermoshier/Engram",
        project_only=True,
        limit=5,
        max_files=10,
    )

    assert report["candidate_count"] == 1
    assert report["project_mismatch_count"] == 0
    candidate = report["candidates"][0]
    assert candidate["path"] == str(transcript)
    assert candidate["session_cwd"] == "/Users/konnermoshier"
    assert candidate["project_match"] is True
    assert candidate["project_match_source"] == "tool_workdir"
    assert {
        "source": "tool_workdir",
        "path": "/Users/konnermoshier/Engram",
    } in candidate["project_evidence_paths"]


def test_build_dogfood_candidate_report_can_sort_by_recent(tmp_path) -> None:
    older_many_turns = tmp_path / "older-many-turns.jsonl"
    recent_few_turns = tmp_path / "recent-few-turns.jsonl"
    older_many_turns.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "older", "cwd": "/tmp/engram"},
                    }
                ),
                *(
                    json.dumps({"role": "user", "content": f"Older Engram turn {index}"})
                    for index in range(4)
                ),
            ]
        ),
        encoding="utf-8",
    )
    recent_few_turns.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {"id": "recent", "cwd": "/tmp/other"},
                    }
                ),
                json.dumps({"role": "user", "content": "Recent resumed dogfood turn"}),
            ]
        ),
        encoding="utf-8",
    )
    os.utime(older_many_turns, (1000, 1000))
    os.utime(recent_few_turns, (2000, 2000))

    by_turns = build_dogfood_candidate_report(
        root_path=tmp_path,
        project_path="/tmp/engram",
        limit=2,
        max_files=10,
    )
    by_recent = build_dogfood_candidate_report(
        root_path=tmp_path,
        project_path="/tmp/engram",
        limit=2,
        max_files=10,
        sort="recent",
    )

    assert by_turns["sort"] == "turns"
    assert by_turns["candidates"][0]["path"] == str(older_many_turns)
    assert by_recent["sort"] == "recent"
    assert by_recent["candidates"][0]["path"] == str(recent_few_turns)


def test_build_dogfood_label_template_uses_hashes_not_content() -> None:
    report = {
        "source": {
            "path": "/tmp/transcript.md",
            "transcript_hash": "abc123",
            "user_turn_count": 1,
        },
        "project_path": "/tmp/engram",
        "group_id": "native_brain",
        "modes": ["off", "gated_lite"],
        "turns": [
            {
                "index": 0,
                "content_hash": "turnhash",
                "content": "Private transcript body",
                "need": {
                    "need_type": "project_state",
                    "should_recall": True,
                    "query_hint": "Engram AXI",
                },
                "decisions": [
                    {
                        "mode": "gated_lite",
                        "decision": "triggered",
                        "mode_executed": "lite",
                        "reason": "memory_need",
                    }
                ],
            }
        ],
    }

    template = build_dogfood_label_template(report)

    assert template["kind"] == "engram.dogfood_label_template.v1"
    assert template["opt_in_required"] is True
    assert template["source"]["content_redacted"] is True
    assert template["source"]["transcript_hash"] == "abc123"
    assert template["turns"][0]["content_hash"] == "turnhash"
    assert template["turns"][0]["labels"]["memory_was_needed"] is None
    assert template["turns"][0]["labels"]["stale_modes"] == []
    assert template["turns"][0]["labels"]["corrected_modes"] == []
    assert template["turns"][0]["query_hint"] is None
    assert template["turns"][0]["query_hint_redacted"] is True
    assert "Private transcript body" not in json.dumps(template)
    assert "Engram AXI" not in json.dumps(template)


def test_build_dogfood_label_template_can_include_content_by_explicit_opt_in() -> None:
    report = {
        "source": {
            "path": "/tmp/transcript.md",
            "transcript_hash": "abc123",
            "user_turn_count": 1,
        },
        "project_path": "/tmp/engram",
        "group_id": "native_brain",
        "modes": ["cached"],
        "turns": [
            {
                "index": 0,
                "content_hash": "turnhash",
                "need": {
                    "need_type": "project_state",
                    "should_recall": True,
                    "query_hint": "Engram AXI",
                },
                "decisions": [
                    {
                        "mode": "cached",
                        "decision": "triggered",
                        "mode_executed": "cached",
                        "reason": "cache_candidate",
                    }
                ],
            }
        ],
    }

    template = build_dogfood_label_template(
        report,
        include_content=True,
        content_lookup={"turnhash": "Private local transcript body"},
    )

    assert template["source"]["content_redacted"] is False
    assert template["turns"][0]["content"] == "Private local transcript body"
    assert template["turns"][0]["query_hint"] == "Engram AXI"
    assert template["turns"][0]["query_hint_redacted"] is False
    assert "explicitly opted in" in template["instructions"][0]


def test_dogfood_label_edit_helpers_update_turns_and_session_samples(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram AXI",
                        "decisions": [
                            {"mode": "cached", "decision": "triggered"},
                            {"mode": "deep", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": None,
                            "best_mode": None,
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "stale_modes": [],
                            "corrected_modes": [],
                            "notes": "",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    turn_report = update_dogfood_turn_label(
        labels_path=labels_path,
        turn_index=0,
        memory_needed=True,
        best_mode="cached",
        helpful_modes=["deep"],
        stale_modes=["deep"],
        notes="cached was enough",
    )
    session_report = add_dogfood_session_label(
        labels_path=labels_path,
        scenario="Dogfood continuity",
        baseline_score=0.2,
        memory_score=0.8,
        open_loop_expected=True,
        open_loop_recovered=True,
        temporal_expected=True,
        temporal_correct=True,
    )
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    recall_samples, session_samples, summary = build_dogfood_label_import_samples(artifact)
    markdown = render_dogfood_label_edit_markdown(session_report)

    assert turn_report["status"] == "updated"
    assert turn_report["labels"]["best_mode"] == "cached"
    assert artifact["turns"][0]["labels"]["helpful_modes"] == ["deep"]
    assert artifact["turns"][0]["labels"]["stale_modes"] == ["deep"]
    assert artifact["session_samples"][0]["scenario"] == "Dogfood continuity"
    assert summary["recall_sample_count"] == 2
    assert summary["session_sample_count"] == 1
    assert len(recall_samples) == 2
    assert len(session_samples) == 1
    assert session_report["review"]["ready"] is True
    assert session_report["review"]["next_commands"]["import_labels"]
    assert session_report["review"]["next_commands"]["export_evidence"]
    assert "# Engram Dogfood Label Edit" in markdown


def test_dogfood_review_suggests_label_commands(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 2,
                        "content_hash": "turnhash",
                        "query_hint": "Need the prior Engram plan",
                        "decisions": [
                            {"mode": "off", "decision": "skipped"},
                            {"mode": "cached", "decision": "triggered"},
                            {"mode": "deep", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": None,
                            "best_mode": None,
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "stale_modes": [],
                            "corrected_modes": [],
                            "notes": "",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_review_report(
        labels_path=labels_path,
        min_recall_samples=1,
        min_session_samples=1,
    )
    markdown = render_dogfood_review_markdown(report)

    turn_command = report["label_commands"]["turns"][0]
    assert turn_command["turn"] == 2
    assert report["label_commands"]["next_turn"] == turn_command
    assert report["review"]["review_queue"][0]["query_hint"] is None
    assert report["review"]["review_queue"][0]["query_hint_redacted"] is True
    assert report["review"]["review_queue_summary"] == {
        "total": 1,
        "by_reason": {"memory_was_needed_missing": 1},
        "by_need_type": {"unknown": 1},
        "redacted_query_hint_count": 1,
        "next_review_turn": {
            "index": 2,
            "content_hash": "turnhash",
            "need_type": None,
            "reasons": ["memory_was_needed_missing"],
            "query_hint_redacted": True,
            "available_modes": ["cached", "deep", "off"],
        },
    }
    assert "engram dogfood inspect-turn" in turn_command["inspect"]
    assert "--include-content" in turn_command["inspect"]
    assert "engram dogfood label-turn" in turn_command["memory_needed"]
    assert "--memory-needed yes" in turn_command["memory_needed"]
    assert "--memory-needed no" in turn_command["memory_not_needed"]
    assert "--notes" not in turn_command["memory_needed"]
    assert "--notes" not in turn_command["memory_not_needed"]
    assert turn_command["notes_hint"] == "Add --notes with a real observed reason if useful."
    assert "engram dogfood label-session" in report["label_commands"]["session"]
    assert "--notes" not in report["label_commands"]["session"]
    assert (
        report["label_commands"]["session_notes_hint"]
        == "Add --notes with real session-level review notes if useful."
    )
    assert report["next_commands"]["import_labels"] is None
    assert report["next_commands"]["export_evidence"] is None
    assert "engram dogfood inspect-turn" in markdown
    assert "## Review Summary" in markdown
    assert "## Recommended Next Turn" in markdown
    assert "## Suggested Turn Label Commands" in markdown
    assert "## Suggested Session Label Command" in markdown
    assert "# If memory was needed:" in markdown
    assert "Add --notes with a real observed reason" in markdown
    assert "Add --notes with real session-level review notes" in markdown
    assert "<why memory helped>" not in markdown
    assert "<why memory was unnecessary or misleading>" not in markdown
    assert "<session-level review notes>" not in markdown
    assert "## Next" not in markdown


def test_dogfood_review_can_focus_commands_by_need_type(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "fact",
                        "need_type": "fact_lookup",
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                    {
                        "index": 1,
                        "content_hash": "loop1",
                        "need_type": "open_loop",
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                    {
                        "index": 2,
                        "content_hash": "loop2",
                        "need_type": "open_loop",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_review_report(
        labels_path=labels_path,
        need_type="open_loop",
        command_limit=1,
    )
    markdown = render_dogfood_review_markdown(report)

    assert report["review"]["review_queue_summary"]["by_need_type"] == {
        "open_loop": 2,
        "fact_lookup": 1,
    }
    commands = report["label_commands"]
    assert commands["need_type_filter"] == "open_loop"
    assert commands["matching_turn_count"] == 2
    assert commands["turn_command_count"] == 1
    assert commands["omitted_turn_command_count"] == 1
    assert commands["next_turn"]["turn"] == 1
    assert commands["turns"][0]["turn"] == 1
    assert "Command focus: open_loop" in markdown
    assert "Suggested turn commands: 1/2" in markdown
    assert "Omitted matching commands: 1" in markdown
    assert "## Focused Review Queue (open_loop)" in markdown
    assert "- Showing 2 of 3 queued turn(s)" in markdown
    assert "- turn 1" in markdown
    assert "- turn 2" in markdown
    assert "- turn 0" not in markdown
    assert "### Turn 1" in markdown
    assert "### Turn 0" not in markdown


def test_dogfood_review_does_not_suggest_session_label_without_turns(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_review_report(
        labels_path=labels_path,
        min_recall_samples=1,
        min_session_samples=1,
    )
    markdown = render_dogfood_review_markdown(report)

    assert report["status"] == "trace_only"
    assert report["review"]["turn_count"] == 0
    assert "no_labelable_turns" in report["failures"]
    assert report["next_commands"]["import_labels"] is None
    assert report["next_commands"]["export_evidence"] is None
    assert report["label_commands"]["turns"] == []
    assert report["label_commands"]["session"] is None
    assert "## Suggested Session Label Command" not in markdown
    assert "no labelable turns" in markdown


def test_dogfood_turn_inspection_uses_local_source_only_on_explicit_content(
    tmp_path,
) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Start the Engram review."}),
                json.dumps({"role": "assistant", "content": "I will inspect it."}),
                json.dumps({"role": "user", "content": "What changed with AXI?"}),
            ]
        ),
        encoding="utf-8",
    )
    user_turns = [
        turn
        for turn in parse_transcript_text(transcript.read_text(encoding="utf-8"), source="x")
        if turn.role == "user"
    ]
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "source": {
                    "path": str(transcript),
                    "transcript_hash": "abc123",
                    "content_redacted": True,
                },
                "turns": [
                    {
                        "index": 0,
                        "content_hash": user_turns[0].content_hash,
                        "need_type": "fact_lookup",
                        "should_recall": True,
                        "query_hint": None,
                        "query_hint_redacted": True,
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                    {
                        "index": 1,
                        "content_hash": user_turns[1].content_hash,
                        "need_type": "project_state",
                        "should_recall": True,
                        "query_hint": None,
                        "query_hint_redacted": True,
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    redacted = build_dogfood_turn_inspection_report(
        labels_path=labels_path,
        turn_index=1,
        context=1,
        include_content=False,
    )
    contentful = build_dogfood_turn_inspection_report(
        labels_path=labels_path,
        content_hash=user_turns[1].content_hash,
        context=1,
        include_content=True,
    )
    next_contentful = build_dogfood_turn_inspection_report(
        labels_path=labels_path,
        next_unreviewed=True,
        need_type="project_state",
        context=1,
        include_content=True,
    )
    markdown = render_dogfood_turn_inspection_markdown(contentful)

    assert redacted["status"] == "ready"
    assert redacted["content_redacted"] is True
    assert "content" not in redacted["context_turns"][0]
    assert contentful["content_redacted"] is False
    assert contentful["target"]["turn"] == 1
    assert contentful["target"]["source_content_hash"] == user_turns[1].content_hash
    assert contentful["context_turns"][0]["role"] == "assistant"
    assert contentful["context_turns"][0]["content"] == "I will inspect it."
    assert contentful["context_turns"][1]["selected"] is True
    assert contentful["context_turns"][1]["content"] == "What changed with AXI?"
    assert contentful["label_command"]["turn"] == 1
    assert "--memory-needed yes" in contentful["label_command"]["memory_needed"]
    assert next_contentful["target"]["turn"] == 1
    assert next_contentful["need_type_filter"] == "project_state"
    assert next_contentful["label_command"]["turn"] == 1
    assert "# Engram Dogfood Turn Inspection" in markdown
    assert "## Suggested Label Commands" in markdown
    assert "# If memory was needed:" in markdown
    assert "What changed with AXI?" in markdown


def test_dogfood_review_can_inline_opted_in_review_packet_content(tmp_path) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Start the Engram review."}),
                json.dumps({"role": "assistant", "content": "I will inspect it."}),
                json.dumps({"role": "user", "content": "What changed with AXI?"}),
            ]
        ),
        encoding="utf-8",
    )
    user_turns = [
        turn
        for turn in parse_transcript_text(transcript.read_text(encoding="utf-8"), source="x")
        if turn.role == "user"
    ]
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "source": {
                    "path": str(transcript),
                    "transcript_hash": "abc123",
                    "content_redacted": True,
                },
                "turns": [
                    {
                        "index": 0,
                        "content_hash": user_turns[0].content_hash,
                        "need_type": "fact_lookup",
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                    {
                        "index": 1,
                        "content_hash": user_turns[1].content_hash,
                        "need_type": "project_state",
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    redacted = build_dogfood_review_report(
        labels_path=labels_path,
        need_type="project_state",
        command_limit=1,
    )
    contentful = build_dogfood_review_report(
        labels_path=labels_path,
        need_type="project_state",
        command_limit=1,
        include_content=True,
        context=1,
    )
    redacted_markdown = render_dogfood_review_markdown(redacted)
    contentful_markdown = render_dogfood_review_markdown(contentful)

    assert redacted["inspection"]["content_redacted"] is True
    assert redacted["inspection_previews"] == []
    assert "What changed with AXI?" not in redacted_markdown
    assert contentful["inspection"]["content_redacted"] is False
    assert contentful["inspection"]["preview_count"] == 1
    assert contentful["inspection_previews"][0]["target"]["turn"] == 1
    assert "## Review Packet" in contentful_markdown
    assert "What changed with AXI?" in contentful_markdown
    assert "# If memory was needed:" in contentful_markdown
    assert "Transcript content is included because `--include-content`" in contentful_markdown


@pytest.mark.asyncio
async def test_dogfood_inspect_turn_cli_prints_local_content_when_opted_in(
    tmp_path,
    capsys,
) -> None:
    transcript = tmp_path / "codex.jsonl"
    transcript.write_text(
        json.dumps({"role": "user", "content": "What changed with AXI?"}),
        encoding="utf-8",
    )
    user_hash = parse_transcript_text(
        transcript.read_text(encoding="utf-8"),
        source="x",
    )[0].content_hash
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "source": {
                    "path": str(transcript),
                    "transcript_hash": "abc123",
                    "content_redacted": True,
                },
                "turns": [
                    {
                        "index": 0,
                        "content_hash": user_hash,
                        "need_type": "project_state",
                        "decisions": [{"mode": "cached", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = await run_dogfood_command(
        SimpleNamespace(
            dogfood_command="inspect-turn",
            labels=labels_path,
            turn=0,
            content_hash=None,
            context=0,
            include_content=True,
            format="json",
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["status"] == "ready"
    assert output["context_turns"][0]["content"] == "What changed with AXI?"


def test_build_dogfood_label_import_samples_uses_opted_in_labels() -> None:
    artifact = {
        "kind": "engram.dogfood_label_template.v1",
        "group_id": "native_brain",
        "source": {"transcript_hash": "abc123", "content_redacted": True},
        "turns": [
            {
                "index": 0,
                "content_hash": "turnhash",
                "query_hint": "Engram AXI",
                "decisions": [
                    {"mode": "off", "decision": "skipped"},
                    {"mode": "gated_lite", "decision": "triggered"},
                    {"mode": "deep", "decision": "triggered"},
                ],
                "labels": {
                    "memory_was_needed": True,
                    "best_mode": "gated_lite",
                    "helpful_modes": ["deep"],
                    "false_recall_modes": [],
                    "stale_modes": ["deep"],
                    "corrected_modes": ["gated_lite"],
                    "notes": "lite was enough",
                },
            },
            {
                "index": 1,
                "content_hash": "unreviewed",
                "query_hint": "Skipped",
                "decisions": [{"mode": "deep", "decision": "triggered"}],
                "labels": {"memory_was_needed": None},
            },
        ],
    }

    recall_samples, session_samples, summary = build_dogfood_label_import_samples(artifact)

    assert summary["group_id"] == "native_brain"
    assert summary["recall_sample_count"] == 2
    assert summary["skipped_turn_count"] == 1
    assert session_samples == []
    assert {sample.source for sample in recall_samples} == {
        "dogfood_review:deep",
        "dogfood_review:gated_lite",
    }
    lite = next(sample for sample in recall_samples if sample.source.endswith("gated_lite"))
    assert lite.recall_needed is True
    assert lite.recall_triggered is True
    assert lite.recall_helped is True
    assert lite.packets_used == 1
    assert lite.corrected_packets == 1
    assert lite.query is None
    assert "turnhash" in (lite.notes or "")
    deep = next(sample for sample in recall_samples if sample.source.endswith("deep"))
    assert deep.stale_packets == 1


def test_dogfood_memory_operation_metrics_from_replay_trace_evidence() -> None:
    report = {
        "trace_evidence": {
            "status": "measured",
            "trace_count": 3,
            "operation_counts": {"home": 1, "context": 1, "recall": 1},
            "status_counts": {"healthy": 1, "ok": 1, "degraded": 1},
            "client_counts": {"codex": 3},
            "origin_counts": {"session-start-hook": 1, "agent-followup": 2},
            "duration_ms": {"avg": 100.0, "p95": 220.0, "max": 220.0},
            "timeout_count": 1,
            "degraded_count": 1,
            "cache_hit_count": 1,
            "packet_count": 4,
            "result_count": 2,
        }
    }

    metrics = dogfood_memory_operation_metrics_from_replay_report(report)

    assert metrics["operation_count"] == 3
    assert metrics["duration_ms"] == {"avg": 100.0, "p95": 220.0}
    assert metrics["operation_counts"] == {"context": 1, "home": 1, "recall": 1}
    assert metrics["source_counts"] == {"codex": 3}
    assert metrics["completed_count"] == 2
    assert metrics["timeout_count"] == 1
    assert metrics["degraded_count"] == 1
    assert metrics["budget_miss_count"] == 1
    assert metrics["cache_hit_count"] == 1
    assert metrics["cache_miss_count"] == 2
    assert metrics["packet_count"] == 4
    assert metrics["result_count"] == 2


@pytest.mark.asyncio
async def test_import_dogfood_replay_cost_metrics_persists_snapshot(tmp_path) -> None:
    replay_path = tmp_path / "replay.json"
    sqlite_path = tmp_path / "engram.db"
    replay_path.write_text(
        json.dumps(
            {
                "source": {"transcript_hash": "abc123"},
                "trace_evidence": {
                    "status": "measured",
                    "trace_count": 2,
                    "operation_counts": {"home": 1, "context": 1},
                    "status_counts": {"healthy": 1, "ok": 1},
                    "origin_counts": {"session-start-hook": 1, "agent-followup": 1},
                    "duration_ms": {"avg": 50.0, "p95": 80.0},
                    "cache_hit_count": 1,
                    "packet_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    report = await import_dogfood_replay_cost_metrics(
        replay_report=replay_path,
        sqlite_path=sqlite_path,
        group_id="native_brain",
    )

    store = SQLiteEvaluationStore(str(sqlite_path))
    await store.initialize()
    try:
        metrics = await store.get_latest_memory_operation_metrics_snapshot("native_brain")
    finally:
        await store.close()

    assert report["status"] == "imported"
    assert report["operation_count"] == 2
    assert metrics["operation_count"] == 2
    assert metrics["duration_ms"]["p95"] == 80.0
    assert metrics["cache_hit_count"] == 1
    assert metrics["packet_count"] == 2


def test_dogfood_review_report_shows_missing_and_invalid_labels(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "ready",
                        "query_hint": "Engram AXI",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    },
                    {
                        "index": 1,
                        "content_hash": "missing",
                        "query_hint": "Needs human review",
                        "decisions": [
                            {"mode": "deep", "decision": "triggered"},
                        ],
                        "labels": {"memory_was_needed": None},
                    },
                    {
                        "index": 2,
                        "content_hash": "invalid",
                        "query_hint": "Bad mode",
                        "decisions": [
                            {"mode": "cached", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "magic",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    },
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity review",
                        "baselineScore": 0.2,
                        "memoryScore": 0.7,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_review_report(labels_path=labels_path)
    markdown = render_dogfood_review_markdown(report)

    assert report["status"] == "invalid_labels"
    assert report["ready"] is False
    assert report["reviewed_labels"]["recall_sample_count"] == 1
    assert report["reviewed_labels"]["session_sample_count"] == 1
    assert report["review"]["turn_count"] == 3
    assert report["review"]["reviewed_turn_count"] == 2
    assert report["review"]["importable_turn_count"] == 1
    assert report["review"]["unreviewed_turn_count"] == 1
    assert report["review"]["invalid_label_count"] == 1
    assert report["review"]["review_queue_summary"]["total"] == 2
    assert report["review"]["review_queue_summary"]["by_reason"] == {
        "memory_was_needed_missing": 1,
        "unreplayed_modes:magic": 1,
        "unsupported_modes:magic": 1,
    }
    assert report["review"]["review_queue_summary"]["by_need_type"] == {"unknown": 2}
    assert "invalid_label_turns(1)" in report["failures"]
    assert "memory_was_needed_missing" in report["review"]["review_queue"][0]["reasons"]
    assert "unsupported_modes:magic" in report["review"]["review_queue"][1]["reasons"]
    assert "unreplayed_modes:magic" in report["review"]["review_queue"][1]["reasons"]
    assert "# Engram Dogfood Label Review" in markdown
    assert "turn 1 (missing): memory_was_needed_missing" in markdown
    assert "Bad mode" not in markdown


def test_dogfood_review_rejects_placeholder_review_text(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram AXI",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "notes": "<why memory helped>",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "<reviewed continuity scenario>",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                        "notes": "<session-level review notes>",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_review_report(labels_path=labels_path)
    markdown = render_dogfood_review_markdown(report)

    assert report["status"] == "invalid_labels"
    assert report["ready"] is False
    assert report["reviewed_labels"]["recall_sample_count"] == 0
    assert report["reviewed_labels"]["session_sample_count"] == 0
    assert "invalid_label_turns(1)" in report["failures"]
    assert "invalid_session_samples(1)" in report["failures"]
    assert "placeholder_label_notes" in report["review"]["review_queue"][0]["reasons"]
    assert report["review"]["invalid_session_samples"][0]["reasons"] == [
        "placeholder_session_scenario",
        "placeholder_session_notes",
    ]
    assert report["next_commands"]["import_labels"] is None
    assert report["next_commands"]["export_evidence"] is None
    assert "placeholder_label_notes" in markdown
    assert "placeholder_session_scenario" in markdown


@pytest.mark.asyncio
async def test_import_dogfood_label_artifact_persists_to_evaluation_store(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram AXI",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                            {"mode": "deep", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": False,
                            "best_mode": None,
                            "helpful_modes": [],
                            "false_recall_modes": ["deep"],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                        "openLoopExpected": True,
                        "openLoopRecovered": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await import_dogfood_label_artifact(
        labels_path=labels_path,
        sqlite_path=sqlite_path,
    )

    assert report["status"] == "imported"
    assert report["recall_sample_count"] == 1
    assert report["session_sample_count"] == 1
    store = SQLiteEvaluationStore(str(sqlite_path))
    await store.initialize()
    try:
        recall_samples = await store.get_recall_samples("native_brain")
        session_samples = await store.get_session_samples("native_brain")
    finally:
        await store.close()

    assert len(recall_samples) == 1
    assert recall_samples[0].recall_needed is False
    assert recall_samples[0].false_recalls == 1
    assert len(session_samples) == 1
    assert session_samples[0].open_loop_recovered is True


@pytest.mark.asyncio
async def test_import_dogfood_label_artifact_is_idempotent(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram AXI",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    await import_dogfood_label_artifact(labels_path=labels_path, sqlite_path=sqlite_path)
    await import_dogfood_label_artifact(labels_path=labels_path, sqlite_path=sqlite_path)

    store = SQLiteEvaluationStore(str(sqlite_path))
    await store.initialize()
    try:
        recall_samples = await store.get_recall_samples("native_brain")
        session_samples = await store.get_session_samples("native_brain")
    finally:
        await store.close()

    assert len(recall_samples) == 1
    assert len(session_samples) == 1


@pytest.mark.asyncio
async def test_import_dogfood_label_artifact_refuses_placeholder_review_text(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "notes": "<why memory helped>",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await import_dogfood_label_artifact(
        labels_path=labels_path,
        sqlite_path=sqlite_path,
    )

    assert report["status"] == "invalid_labels"
    assert report["ready"] is False
    assert "invalid_label_turns(1)" in report["failures"]
    assert not sqlite_path.exists()


def test_build_dogfood_human_label_evidence_artifact_is_gate_compatible() -> None:
    labels = {
        "kind": "engram.dogfood_label_template.v1",
        "group_id": "native_brain",
        "source": {"transcript_hash": "abc123", "content_redacted": True},
        "turns": [
            {
                "index": 0,
                "content_hash": "turnhash",
                "query_hint": "Engram AXI value report",
                "decisions": [
                    {"mode": "gated_lite", "decision": "triggered"},
                    {"mode": "deep", "decision": "triggered"},
                ],
                "labels": {
                    "memory_was_needed": True,
                    "best_mode": "gated_lite",
                    "helpful_modes": ["deep"],
                    "false_recall_modes": [],
                    "stale_modes": ["deep"],
                    "corrected_modes": ["gated_lite"],
                    "notes": "reviewed against the original local transcript",
                },
            }
        ],
        "session_samples": [
            {
                "scenario": "Codex dogfood continuity review",
                "baselineScore": 0.1,
                "memoryScore": 0.9,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": True,
                "notes": "memory preserved the active Engram goal",
            }
        ],
    }

    artifact, summary = build_dogfood_human_label_evidence_artifact(
        labels,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )
    assert summary["recall_sample_count"] == 2
    assert artifact["kind"] == "engram_human_label_evidence"
    assert artifact["dogfood"]["transcriptHash"] == "abc123"
    assert evidence["status"] == "measured"
    assert evidence["sample_sources"] == ["native_dogfood_harness"]
    lite = next(
        sample
        for sample in artifact["recallSamples"]
        if sample["notes"].endswith("mode=gated_lite")
    )
    assert lite["correctedPackets"] == 1
    deep = next(
        sample for sample in artifact["recallSamples"] if sample["notes"].endswith("mode=deep")
    )
    assert deep["stalePackets"] == 1
    assert "Private transcript body" not in json.dumps(artifact)


def test_export_dogfood_human_label_evidence_writes_validation_artifact(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [
                            {"mode": "off", "decision": "skipped"},
                        ],
                        "labels": {
                            "memory_was_needed": False,
                            "best_mode": "off",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "notes": "no memory needed",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity review",
                        "baselineScore": 0.5,
                        "memoryScore": 0.5,
                        "temporalExpected": False,
                        "temporalCorrect": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = export_dogfood_human_label_evidence(
        labels_path=labels_path,
        output_path=evidence_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )

    artifact = json.loads(evidence_path.read_text(encoding="utf-8"))
    markdown = render_dogfood_export_markdown(report)
    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=1,
        min_session_samples=1,
    )
    assert report["status"] == "exported"
    assert report["ready"] is True
    assert evidence["status"] == "measured"
    assert artifact["recallSamples"][0]["query"] == "dogfood reviewed turn 0 hash turnhash"
    assert "--human-label-artifact" in report["validation_command"]
    assert "# Engram Dogfood Evidence Export" in markdown


def test_export_dogfood_human_label_evidence_refuses_placeholder_metadata(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [
                            {"mode": "off", "decision": "skipped"},
                        ],
                        "labels": {
                            "memory_was_needed": False,
                            "best_mode": "off",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "notes": "no memory needed",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity review",
                        "baselineScore": 0.5,
                        "memoryScore": 0.5,
                        "temporalExpected": False,
                        "temporalCorrect": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = export_dogfood_human_label_evidence(
        labels_path=labels_path,
        output_path=evidence_path,
        source="native_dogfood_harness",
        client="<client>",
        captured_at="<ISO-8601>",
        labeler="<human-reviewer>",
    )
    markdown = render_dogfood_export_markdown(report)

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert "human_label_evidence:placeholder_harness_client" in report["failures"]
    assert "human_label_evidence:placeholder_harness_captured_at" in report["failures"]
    assert "human_label_evidence:placeholder_human_labeler" in report["failures"]
    assert not evidence_path.exists()
    assert "No evidence artifact was written" in markdown
    assert "placeholder_harness_client" in markdown


def test_export_dogfood_human_label_evidence_refuses_placeholder_review_text(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                            "notes": "dogfood notes copied from <why memory helped>",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity review",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = export_dogfood_human_label_evidence(
        labels_path=labels_path,
        output_path=evidence_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )
    markdown = render_dogfood_export_markdown(report)

    assert report["status"] == "failed"
    assert report["ready"] is False
    assert "invalid_label_turns(1)" in report["failures"]
    assert "placeholder_label_notes" in report["review"]["review_queue"][0]["reasons"]
    assert not evidence_path.exists()
    assert "Replace placeholder review text" in markdown


@pytest.mark.asyncio
async def test_export_evidence_cli_returns_nonzero_for_invalid_metadata(
    tmp_path,
    capsys,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [{"mode": "off", "decision": "skipped"}],
                        "labels": {
                            "memory_was_needed": False,
                            "best_mode": "off",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Dogfood continuity review",
                        "baselineScore": 0.5,
                        "memoryScore": 0.5,
                        "temporalExpected": False,
                        "temporalCorrect": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = await run_dogfood_command(
        SimpleNamespace(
            dogfood_command="export-evidence",
            labels=labels_path,
            out=evidence_path,
            source="native_dogfood_harness",
            client="<client>",
            captured_at="<ISO-8601>",
            labeler="<human-reviewer>",
            session_id=None,
            group_id=None,
            include_all_modes=False,
            format="json",
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["status"] == "failed"
    assert not evidence_path.exists()


def test_dogfood_closeout_report_builds_native_memory_value_commands(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram native dogfood",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    export_dogfood_human_label_evidence(
        labels_path=labels_path,
        output_path=evidence_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )

    report = build_dogfood_closeout_report(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
        sqlite_path=sqlite_path,
        group_id="native_brain",
        helix_data_dir=tmp_path / "helix",
    )
    markdown = render_dogfood_closeout_markdown(report)

    assert report["status"] == "ready_for_native_memory_value"
    assert report["ready"] is True
    assert report["failures"] == []
    assert report["reviewed_labels"]["recall_sample_count"] == 1
    assert report["reviewed_labels"]["session_sample_count"] == 1
    assert report["human_label_evidence"]["status"] == "measured"
    assert (
        "ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native"
        in report["commands"]["native_memory_value"]
    )
    assert "--require-memory-value" in report["commands"]["native_memory_value"]
    assert "--sqlite-path" in report["commands"]["import_labels"]
    assert "<human-reviewer>" in report["commands"]["export_evidence"]
    assert "# Engram Dogfood Closeout" in markdown


def test_dogfood_closeout_report_keeps_missing_evidence_incomplete(tmp_path) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_closeout_report(labels_path=labels_path)

    assert report["status"] == "needs_evidence"
    assert report["ready"] is False
    assert "reviewed_recall_samples(0<1)" in report["failures"]
    assert "reviewed_session_samples(0<1)" in report["failures"]
    assert "missing_human_label_artifact" in report["failures"]
    assert report["reviewed_labels"]["status"] == "needs_labels"
    assert report["human_label_evidence"]["status"] == "missing"
    assert report["commands"]["import_labels"] is None
    assert report["commands"]["export_evidence"] is None
    assert report["commands"]["native_memory_value"] is None
    markdown = render_dogfood_closeout_markdown(report)
    assert "No closeout command is suggested" in markdown
    assert "engram dogfood import-labels" not in markdown
    assert any(
        "Replay decisions and synthetic benchmark artifacts do not satisfy" in note
        for note in report["notes"]
    )


def test_dogfood_closeout_report_treats_missing_evidence_path_as_missing(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "deep",
                            "helpful_modes": ["deep"],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_closeout_report(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
    )

    assert report["status"] == "needs_evidence"
    assert report["ready"] is False
    assert "missing_human_label_artifact" in report["failures"]
    assert not any(
        failure.startswith("invalid_human_label_artifact") for failure in report["failures"]
    )
    assert report["reviewed_labels"]["status"] == "measured"
    assert report["human_label_evidence"] == {
        "status": "missing",
        "artifact_path": str(evidence_path),
        "failures": ["missing_human_label_artifact"],
    }
    assert report["commands"]["import_labels"].startswith("engram dogfood import-labels")
    assert report["commands"]["export_evidence"].startswith("engram dogfood export-evidence")
    assert report["commands"]["native_memory_value"] is None


def test_dogfood_closeout_report_stages_commands_after_reviewed_labels(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "deep",
                            "helpful_modes": ["deep"],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_dogfood_closeout_report(
        labels_path=labels_path,
        sqlite_path=sqlite_path,
    )
    markdown = render_dogfood_closeout_markdown(report)

    assert report["status"] == "needs_evidence"
    assert report["reviewed_labels"]["status"] == "measured"
    assert report["human_label_evidence"]["status"] == "missing"
    assert report["commands"]["import_labels"].startswith("engram dogfood import-labels")
    assert report["commands"]["export_evidence"].startswith("engram dogfood export-evidence")
    assert report["commands"]["native_memory_value"] is None
    assert "engram dogfood import-labels" in markdown
    assert "engram dogfood export-evidence" in markdown
    assert "engram evaluate --mode helix" not in markdown


@pytest.mark.asyncio
async def test_finalize_dogfood_labels_imports_exports_and_runs_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    replay_path = tmp_path / "dogfood-replay.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram native dogfood",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": [],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    replay_path.write_text(
        json.dumps(
            {
                "source": {"transcript_hash": "abc123"},
                "trace_evidence": {
                    "status": "measured",
                    "trace_count": 1,
                    "operation_counts": {"context": 1},
                    "status_counts": {"ok": 1},
                    "duration_ms": {"avg": 32.0, "p95": 32.0},
                    "cache_hit_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    calls = []

    async def fake_evaluate(**kwargs):
        calls.append(kwargs)
        return {
            "status": "measured",
            "exit_code": 0,
            "command": "engram evaluate --memory-value",
            "memory_value": {"status": "measured"},
        }

    monkeypatch.setattr(
        "engram.evaluation.dogfood._run_dogfood_memory_value_evaluation",
        fake_evaluate,
    )

    report = await finalize_dogfood_labels(
        labels_path=labels_path,
        replay_report=replay_path,
        human_label_artifact=evidence_path,
        sqlite_path=sqlite_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
        group_id="native_brain",
        helix_data_dir=tmp_path / "helix",
    )
    markdown = render_dogfood_finalize_markdown(report)

    assert report["status"] == "finalized"
    assert report["ready"] is True
    assert report["import"]["status"] == "imported"
    assert report["cost"]["status"] == "imported"
    assert report["cost"]["operation_count"] == 1
    assert report["export"]["status"] == "exported"
    assert report["closeout"]["ready"] is True
    assert report["evaluation"]["status"] == "measured"
    assert calls[0]["mode"] == "helix"
    assert calls[0]["human_label_artifact"] == evidence_path
    assert evidence_path.exists()
    assert "# Engram Dogfood Finalize" in markdown

    store = SQLiteEvaluationStore(str(sqlite_path))
    await store.initialize()
    try:
        assert len(await store.get_recall_samples("native_brain")) == 1
        assert len(await store.get_session_samples("native_brain")) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_finalize_dogfood_labels_stops_before_import_for_placeholder_metadata(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram native dogfood",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": ["gated_lite"],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await finalize_dogfood_labels(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
        sqlite_path=sqlite_path,
        source="native_dogfood_harness",
        client="<client>",
        captured_at="<ISO-8601>",
        labeler="<human-reviewer>",
        group_id="native_brain",
        helix_data_dir=tmp_path / "helix",
    )

    assert report["status"] == "needs_evidence"
    assert report["ready"] is False
    assert report["phase"] == "evidence_preflight"
    assert "import" not in report
    assert "export" not in report
    assert "closeout" not in report
    assert "human_label_evidence:placeholder_harness_client" in report["failures"]
    assert "human_label_evidence:placeholder_harness_captured_at" in report["failures"]
    assert "human_label_evidence:placeholder_human_labeler" in report["failures"]
    assert not evidence_path.exists()
    assert not sqlite_path.exists()


@pytest.mark.asyncio
async def test_finalize_dogfood_labels_skip_evaluate_requires_manual_gate(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "query_hint": "Engram native dogfood",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": ["gated_lite"],
                            "false_recall_modes": [],
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await finalize_dogfood_labels(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
        sqlite_path=sqlite_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
        group_id="native_brain",
        helix_data_dir=tmp_path / "helix",
        skip_evaluate=True,
    )
    markdown = render_dogfood_finalize_markdown(report)

    assert report["status"] == "needs_evaluation"
    assert report["ready"] is False
    assert report["phase"] == "evaluation"
    assert report["import"]["status"] == "imported"
    assert report["export"]["status"] == "exported"
    assert report["closeout"]["ready"] is True
    assert report["evaluation"]["status"] == "skipped"
    assert report["failures"] == ["native_memory_value_evaluation_skipped"]
    assert report["manual_evaluation_command"] == report["evaluation"]["command"]
    assert "ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native" in report["manual_evaluation_command"]
    assert "## Manual Evaluation Required" in markdown
    assert evidence_path.exists()


@pytest.mark.asyncio
async def test_finalize_dogfood_labels_stops_before_import_when_review_incomplete(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await finalize_dogfood_labels(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )

    assert report["status"] == "needs_labels"
    assert report["ready"] is False
    assert report["phase"] == "review"
    assert not evidence_path.exists()


@pytest.mark.asyncio
async def test_finalize_dogfood_labels_stops_before_import_for_placeholder_review_text(
    tmp_path,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    evidence_path = tmp_path / "human-labels.json"
    sqlite_path = tmp_path / "engram.db"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "turnhash",
                        "decisions": [
                            {"mode": "gated_lite", "decision": "triggered"},
                        ],
                        "labels": {
                            "memory_was_needed": True,
                            "best_mode": "gated_lite",
                            "helpful_modes": ["gated_lite"],
                            "false_recall_modes": [],
                            "notes": "<why memory helped>",
                        },
                    }
                ],
                "session_samples": [
                    {
                        "scenario": "Native dogfood continuity",
                        "baselineScore": 0.2,
                        "memoryScore": 0.8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = await finalize_dogfood_labels(
        labels_path=labels_path,
        human_label_artifact=evidence_path,
        sqlite_path=sqlite_path,
        source="native_dogfood_harness",
        client="Codex",
        captured_at="2026-05-21T18:00:00Z",
        labeler="operator",
    )

    assert report["status"] == "needs_labels"
    assert report["ready"] is False
    assert report["phase"] == "review"
    assert "invalid_label_turns(1)" in report["failures"]
    assert "import" not in report
    assert not evidence_path.exists()
    assert not sqlite_path.exists()


@pytest.mark.asyncio
async def test_dogfood_closeout_require_ready_returns_nonzero_when_incomplete(
    tmp_path,
    capsys,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [],
            }
        ),
        encoding="utf-8",
    )

    exit_code = await run_dogfood_command(
        SimpleNamespace(
            dogfood_command="closeout",
            labels=labels_path,
            human_label_artifact=None,
            sqlite_path=None,
            group_id=None,
            mode="helix",
            helix_data_dir=None,
            min_recall_samples=1,
            min_session_samples=1,
            include_all_modes=False,
            require_ready=True,
            format="json",
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["ready"] is False
    assert "missing_human_label_artifact" in output["failures"]
    assert output["commands"]["import_labels"] is None
    assert output["commands"]["export_evidence"] is None
    assert output["commands"]["native_memory_value"] is None


@pytest.mark.asyncio
async def test_dogfood_review_require_ready_returns_nonzero_when_labels_need_work(
    tmp_path,
    capsys,
) -> None:
    labels_path = tmp_path / "dogfood-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "kind": "engram.dogfood_label_template.v1",
                "group_id": "native_brain",
                "source": {"transcript_hash": "abc123", "content_redacted": True},
                "turns": [
                    {
                        "index": 0,
                        "content_hash": "unreviewed",
                        "decisions": [{"mode": "deep", "decision": "triggered"}],
                        "labels": {"memory_was_needed": None},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = await run_dogfood_command(
        SimpleNamespace(
            dogfood_command="review",
            labels=labels_path,
            group_id=None,
            min_recall_samples=1,
            min_session_samples=1,
            include_all_modes=False,
            require_ready=True,
            format="json",
        )
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert output["ready"] is False
    assert output["status"] == "needs_labels"
    assert "reviewed_recall_samples(0<1)" in output["failures"]
