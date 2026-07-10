from __future__ import annotations

import json

from engram.retrieval.chat_events import (
    accumulate_chat_tool_result,
    build_chat_tool_events,
    build_chat_tool_result_message,
    build_chat_tool_stream_events,
    raw_recall_from_chat_item,
)


def test_build_chat_tool_events_from_recall_and_facts() -> None:
    recall_results = [
        {
            "result_type": "entity",
            "entity": {
                "id": "ent_alice",
                "name": "Alice",
                "type": "Person",
                "summary": "Works on Engram.",
            },
            "score": 0.81234,
            "score_breakdown": {"activation": 0.4567},
            "relationships": [
                {
                    "source_id": "ent_alice",
                    "target_id": "ent_engram",
                    "predicate": "WORKS_ON",
                    "weight": 1.0,
                },
                {
                    "source_id": "ent_alice",
                    "target_id": "ent_native",
                    "predicate": "PREFERS",
                    "weight": 0.9,
                },
                {
                    "source_id": "ent_alice",
                    "target_id": "ent_eval",
                    "predicate": "TRACKS",
                    "weight": 0.7,
                },
            ],
        },
        {
            "result_type": "entity",
            "entity": {"id": "ent_engram", "name": "Engram", "type": "Project"},
            "score": 0.7,
            "score_breakdown": {"activation": 0.3},
            "relationships": [],
        },
        {
            "result_type": "entity",
            "entity": {"id": "ent_native", "name": "Native Helix", "type": "Technology"},
            "score": 0.6,
            "score_breakdown": {"activation": 0.5},
            "relationships": [],
        },
        {
            "result_type": "episode",
            "episode": {
                "id": "ep_1",
                "content": "Discussed PyO3 native mode.",
                "source": "chat",
                "created_at": "2026-05-15T12:00:00",
            },
            "score": 0.51,
        },
        {
            "result_type": "cue_episode",
            "cue": {"episode_id": "ep_cue", "cue_text": "latent native cue"},
            "episode": {"id": "ep_cue", "source": "observe", "created_at": None},
            "score": 0.49,
        },
    ]
    facts = [
        {
            "subject": "Alice",
            "predicate": "WORKS_ON",
            "object": "Engram",
            "confidence": 0.92,
        }
    ]

    events = build_chat_tool_events(recall_results, facts)

    assert [event.name for event in events] == [
        "show_entities",
        "show_relationship_graph",
        "show_facts",
        "show_activation_chart",
        "show_timeline",
    ]
    assert events[0].input["entities"][0] == {
        "id": "ent_alice",
        "name": "Alice",
        "entityType": "Person",
        "summary": "Works on Engram.",
        "score": 0.812,
        "activation": 0.457,
    }
    assert events[1].input["centralEntity"] == "Alice"
    assert events[1].input["nodes"][1]["name"] == "Engram"
    assert events[2].input["facts"][0]["object"] == "Engram"
    assert events[3].input["entities"][0]["name"] == "Native Helix"
    assert events[4].input["episodes"][1] == {
        "id": "ep_cue",
        "content": "latent native cue",
        "source": "observe",
        "createdAt": None,
        "score": 0.49,
        "latent": True,
    }


def test_build_chat_tool_stream_events_pairs_input_and_output_payloads() -> None:
    payloads = build_chat_tool_stream_events(
        [],
        [
            {
                "subject": "Alice",
                "predicate": "WORKS_ON",
                "object": "Engram",
                "confidence": 0.92,
            }
        ],
    )

    assert payloads == [
        {
            "type": "tool-input-available",
            "toolCallId": "tc_1",
            "toolName": "show_facts",
            "input": {
                "facts": [
                    {
                        "subject": "Alice",
                        "predicate": "WORKS_ON",
                        "object": "Engram",
                        "confidence": 0.92,
                    }
                ]
            },
            "dynamic": True,
        },
        {
            "type": "tool-output-available",
            "toolCallId": "tc_1",
            "output": "displayed",
            "dynamic": True,
        },
    ]


def test_raw_recall_from_chat_item_round_trips_chat_shapes() -> None:
    entity = raw_recall_from_chat_item(
        {
            "type": "entity",
            "id": "ent_alice",
            "name": "Alice",
            "entityType": "Person",
            "summary": "Works on Engram.",
            "score": 0.812,
            "activation": 0.457,
            "relationships": [
                {
                    "predicate": "WORKS_ON",
                    "source": "Alice",
                    "target": "Engram",
                    "polarity": "positive",
                }
            ],
        }
    )
    cue = raw_recall_from_chat_item(
        {
            "type": "cue_episode",
            "episodeId": "ep_cue",
            "cueText": "latent native cue",
            "supportingSpans": ["native"],
            "projectionState": "cue_only",
            "source": "observe",
            "score": 0.49,
        }
    )
    episode = raw_recall_from_chat_item(
        {
            "type": "episode",
            "content": "Discussed PyO3 native mode.",
            "source": "chat",
            "score": 0.51,
        }
    )

    assert entity["result_type"] == "entity"
    assert entity["entity"]["id"] == "ent_alice"
    assert entity["score_breakdown"]["activation"] == 0.457
    assert entity["relationships"][0]["target_id"] == "Engram"
    assert cue["result_type"] == "cue_episode"
    assert cue["cue"]["supporting_spans"] == ["native"]
    assert episode["result_type"] == "episode"
    assert episode["episode"]["content"] == "Discussed PyO3 native mode."


def test_build_chat_tool_result_message_preserves_tool_call_contract() -> None:
    assert build_chat_tool_result_message("tool_1", '{"ok": true}') == {
        "type": "tool_result",
        "tool_use_id": "tool_1",
        "content": '{"ok": true}',
    }


def test_accumulate_chat_tool_result_collects_recall_and_facts() -> None:
    recall_results: list[dict] = []
    facts: list[dict] = []

    accumulate_chat_tool_result(
        tool_name="recall",
        result=json.dumps(
            {
                "results": [
                    {
                        "type": "cue_episode",
                        "episodeId": "ep_cue",
                        "cueText": "latent native cue",
                        "score": 0.49,
                    }
                ]
            }
        ),
        recall_results=recall_results,
        facts=facts,
    )
    accumulate_chat_tool_result(
        tool_name="search_facts",
        result=json.dumps(
            {
                "facts": [
                    {
                        "subject": "Alice",
                        "predicate": "WORKS_ON",
                        "object": "Engram",
                    }
                ]
            }
        ),
        recall_results=recall_results,
        facts=facts,
    )

    assert recall_results[0]["result_type"] == "cue_episode"
    assert recall_results[0]["cue"]["episode_id"] == "ep_cue"
    assert facts == [
        {
            "subject": "Alice",
            "predicate": "WORKS_ON",
            "object": "Engram",
        }
    ]


def test_accumulate_chat_tool_result_ignores_malformed_json() -> None:
    recall_results: list[dict] = []
    facts: list[dict] = []

    accumulate_chat_tool_result(
        tool_name="recall",
        result="not json",
        recall_results=recall_results,
        facts=facts,
    )

    assert recall_results == []
    assert facts == []
