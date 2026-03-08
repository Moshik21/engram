"""Tests for recall packet assembly."""

from __future__ import annotations

import pytest

from engram.models.recall import MemoryNeed
from engram.retrieval.packets import assemble_memory_packets


@pytest.mark.asyncio
class TestRecallPackets:
    async def test_builds_state_packet_for_project_state(self):
        results = [
            {
                "entity": {
                    "id": "ent_auth",
                    "name": "Auth Migration",
                    "type": "Project",
                    "summary": "Migrating authentication to the new provider.",
                },
                "score": 0.82,
                "score_breakdown": {"planner_support": 0.4},
                "relationships": [
                    {
                        "id": "rel_1",
                        "predicate": "BLOCKED_BY",
                        "source_id": "ent_auth",
                        "target_id": "ent_oauth",
                    }
                ],
                "supporting_intents": ["direct", "topic"],
            }
        ]
        need = MemoryNeed(
            need_type="project_state",
            should_recall=True,
            confidence=0.8,
        )

        packets = await assemble_memory_packets(
            results,
            "How's the auth migration going?",
            mode="explicit_recall",
            memory_need=need,
            resolve_entity_name=lambda entity_id: _resolve_name(
                {"ent_auth": "Auth Migration", "ent_oauth": "OAuth Rollout"},
                entity_id,
            ),
        )

        assert len(packets) == 1
        packet = packets[0]
        assert packet.packet_type == "state_packet"
        assert packet.entity_ids == ["ent_auth"]
        assert "blocked by OAuth Rollout" in packet.evidence_lines[0]

    async def test_builds_open_loop_packet(self):
        results = [
            {
                "entity": {
                    "id": "ent_redis",
                    "name": "Redis Cache Decision",
                    "type": "Project",
                    "summary": "Still deciding whether Redis stays in the architecture.",
                },
                "score": 0.75,
                "score_breakdown": {"planner_support": 0.35},
                "relationships": [],
                "supporting_intents": ["direct"],
            }
        ]
        need = MemoryNeed(
            need_type="open_loop",
            should_recall=True,
            confidence=0.85,
        )

        packets = await assemble_memory_packets(
            results,
            "Did we decide on Redis yet?",
            memory_need=need,
        )

        assert packets[0].packet_type == "open_loop_packet"
        assert packets[0].summary.startswith("Pending thread around Redis Cache Decision")

    async def test_marks_negative_relationship_lines_as_negated(self):
        results = [
            {
                "entity": {
                    "id": "ent_falcon",
                    "name": "Falcon Dashboard",
                    "type": "Project",
                    "summary": "Customer analytics UI.",
                },
                "score": 0.81,
                "score_breakdown": {"planner_support": 0.3},
                "relationships": [
                    {
                        "id": "rel_neg",
                        "predicate": "USES",
                        "source_id": "ent_falcon",
                        "target_id": "ent_react",
                        "polarity": "negative",
                    }
                ],
                "supporting_intents": ["direct"],
            }
        ]

        packets = await assemble_memory_packets(
            results,
            "Which framework does Falcon Dashboard use now?",
            memory_need=MemoryNeed(
                need_type="project_state",
                should_recall=True,
                confidence=0.8,
            ),
            resolve_entity_name=lambda entity_id: _resolve_name(
                {"ent_falcon": "Falcon Dashboard", "ent_react": "React"},
                entity_id,
            ),
        )

        assert len(packets) == 1
        assert any(
            line.startswith("Negated: Falcon Dashboard uses React")
            for line in packets[0].evidence_lines
        )

    async def test_builds_timeline_packet_from_episodes(self):
        results = [
            {
                "result_type": "episode",
                "episode": {
                    "id": "ep_1",
                    "content": "We decided to pause the rollout until auth stabilizes.",
                    "created_at": "2026-03-01T10:00:00",
                },
                "score": 0.74,
            },
            {
                "result_type": "episode",
                "episode": {
                    "id": "ep_2",
                    "content": "Later we revisited the rollout after fixing the token bug.",
                    "created_at": "2026-03-03T15:00:00",
                },
                "score": 0.71,
            },
        ]
        need = MemoryNeed(
            need_type="temporal_update",
            should_recall=True,
            confidence=0.82,
        )

        packets = await assemble_memory_packets(
            results,
            "What changed since last time?",
            memory_need=need,
        )

        assert len(packets) == 1
        assert packets[0].packet_type == "timeline_packet"
        assert packets[0].episode_ids == ["ep_1", "ep_2"]

    async def test_builds_cue_packet_from_latent_episode(self):
        results = [
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_cue_1",
                    "cue_text": "Phoenix redesign discussion with extraction and recall tradeoffs",
                    "supporting_spans": [
                        "What good is storing all the data if the AI agent never uses it?"
                    ],
                    "projection_state": "cue_only",
                },
                "episode": {
                    "id": "ep_cue_1",
                    "source": "mcp",
                    "created_at": "2026-03-05T12:00:00",
                },
                "score": 0.68,
            }
        ]

        packets = await assemble_memory_packets(
            results,
            "What did we say about recall?",
            memory_need=MemoryNeed(
                need_type="broad_context",
                should_recall=True,
                confidence=0.8,
            ),
        )

        assert len(packets) == 1
        assert packets[0].packet_type == "cue_packet"
        assert packets[0].episode_ids == ["ep_cue_1"]
        assert "Phoenix redesign" in packets[0].summary


async def _resolve_name(
    names: dict[str, str],
    entity_id: str,
) -> str:
    return names.get(entity_id, entity_id)
