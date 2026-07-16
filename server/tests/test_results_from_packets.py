"""Naive clients must not see empty results when packets carry hits."""

from __future__ import annotations

from engram.retrieval.presenter import (
    present_api_recall_response,
    present_mcp_recall_response,
    results_from_packets,
)


def test_results_from_packets_mirrors_entity_and_episode():
    packets = [
        {
            "packet_type": "fact_packet",
            "title": "Decision: Ship public MCP",
            "summary": "Public surface freezes to 9 tools.",
            "entity_ids": ["ent_decision_1"],
            "score": 0.9,
        },
        {
            "packet_type": "episode_packet",
            "title": "Episode: install notes",
            "summary": "Helix native install steps.",
            "episode_ids": ["ep_1"],
            "provenance": ["episode:ep_1"],
        },
    ]
    results = results_from_packets(packets)
    types = {r["result_type"] for r in results}
    assert "entity" in types
    assert "episode" in types
    entity = next(r for r in results if r["result_type"] == "entity")
    assert entity["entity"]["id"] == "ent_decision_1"
    assert "Ship public MCP" in entity["entity"]["name"]


def test_present_mcp_mirrors_results_when_empty_but_packets_present():
    packets = [
        {
            "packet_type": "fact_packet",
            "title": "Decision: Continuity first",
            "summary": "Warm path after restart.",
            "entity_ids": ["ent_dec"],
            "provenance": ["entity:ent_dec"],
        }
    ]
    payload = present_mcp_recall_response(query="Decision continuity", results=[], packets=packets)
    assert payload["packets"]
    assert payload["results"], "results must not be empty when packets have content"
    assert payload["lifecycle"]["result_count"] == len(payload["results"])
    assert payload["results_source"] == "packets"
    assert payload["packets_authoritative"] is True
    assert "results_note" in payload


def test_present_api_mirrors_items_when_empty_but_packets_present():
    packets = [
        {
            "packet_type": "episode_packet",
            "title": "Episode: native Helix",
            "summary": "Stored memory covers native install.",
            "episode_ids": ["ep_native"],
        }
    ]
    payload = present_api_recall_response(query="native Helix", results=[], packets=packets)
    assert payload["packets"]
    assert payload["items"], "items must not be empty when packets have content"
    assert payload["lifecycle"]["resultCount"] == len(payload["items"])
    assert payload["resultsSource"] == "packets"
    assert payload["packetsAuthoritative"] is True


def test_present_does_not_double_mirror_when_results_exist():
    results = [
        {
            "result_type": "entity",
            "entity": {"id": "e1", "name": "Alpha", "entity_type": "Concept"},
            "score": 0.5,
        }
    ]
    packets = [
        {
            "packet_type": "fact_packet",
            "title": "Fact: Alpha",
            "entity_ids": ["e1"],
        }
    ]
    payload = present_mcp_recall_response(query="Alpha", results=results, packets=packets)
    assert len(payload["results"]) == 1
    assert "results_source" not in payload
