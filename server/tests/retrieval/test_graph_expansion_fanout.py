"""Ticket 2: the graph-expansion stage must not overrun its own substage cap.

Why this matters more than an expansion: ``graph_expand_timeout`` is one of the
two probes that arm the recall graph gate (``recall_graph_gate``), so when this
stage runs out of budget, the entity-query pool, the graph pool, spreading
activation and entity attributes all stop for the rest of the request.

Measured live (2026-07-24, warm helix shell, quiet profile, 4 A/B pairs):
an all-lowercase query extracts no terms and returns in ~0.03 ms having read
nothing (gate armed 0/4), while the same question with ONE capitalised token
sent the stage over its 75 ms cap and armed the gate on 3/4 -- and every one of
those recorded 4 gated stages skipped. The query class most likely to need the
graph is the one that loses it.

The tests below are ISOLATED (a latency-stubbed store, not the live brain), and
they assert three separate properties, because a probe that only checked
"finished in time" would pass on a stage that finished by doing nothing:

1. the deadline is respected;
2. the reads that DID complete are used, not discarded (the cancel-and-discard
   bug class);
3. the fan-out is bounded and duplicate reads are gone.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from engram.retrieval.graph_expansion import (
    _EXPANSION_FANOUT,
    expand_query_from_graph,
)

pytestmark = pytest.mark.asyncio


@dataclass
class _Entity:
    id: str
    name: str
    summary: str = ""


@dataclass
class _Rel:
    source_id: str
    target_id: str
    predicate: str


class _SlowGraphStore:
    """Graph store whose every read costs a fixed wall-clock delay.

    Records the call log and the peak number of concurrent in-flight reads, so
    "bounded fan-out" is a measurement rather than a claim.
    """

    def __init__(self, *, read_seconds: float = 0.01, entities_per_term: int = 2):
        self.read_seconds = read_seconds
        self.entities_per_term = entities_per_term
        self.calls: list[str] = []
        self.inflight = 0
        self.peak_inflight = 0

    async def _read(self, label: str):
        self.calls.append(label)
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            await asyncio.sleep(self.read_seconds)
        finally:
            self.inflight -= 1

    async def find_entity_candidates(self, term: str, group_id: str):
        await self._read(f"candidates:{term}")
        return [
            _Entity(id=f"ent_{term}_{i}", name=f"Name{term}{i}", summary="")
            for i in range(self.entities_per_term)
        ]

    async def get_relationships(self, entity_id: str, group_id: str | None = None):
        await self._read(f"rels:{entity_id}")
        return [
            _Rel(source_id=entity_id, target_id=f"{entity_id}_n{i}", predicate="RELATES_TO")
            for i in range(3)
        ]

    async def get_entity(self, entity_id: str, group_id: str):
        await self._read(f"entity:{entity_id}")
        return _Entity(id=entity_id, name=f"Neighbor {entity_id}")


# Five capitalised tokens -> five terms, i.e. the shape that blew the live cap.
QUERY = "what about Helix and Konner and Engram and Anthropic and Claude"


class TestExpansionRespectsItsDeadline:
    async def test_deadline_stops_the_stage_before_the_caller_has_to(self):
        store = _SlowGraphStore(read_seconds=0.01)
        stats: dict[str, float] = {}
        started = time.perf_counter()
        await expand_query_from_graph(
            QUERY,
            store,
            "default",
            deadline_seconds=0.05,
            stats_out=stats,
        )
        elapsed = time.perf_counter() - started

        assert elapsed < 0.20, f"stage ran {elapsed * 1000:.0f}ms past a 50ms deadline"
        assert stats["truncated"] == 1.0, "deadline was never reached; fixture is too fast"

    async def test_completed_reads_are_used_not_discarded(self):
        """The point of the deadline: a partial expansion, not an empty one."""
        store = _SlowGraphStore(read_seconds=0.01)
        stats: dict[str, float] = {}
        expanded = await expand_query_from_graph(
            QUERY,
            store,
            "default",
            deadline_seconds=0.05,
            stats_out=stats,
        )

        assert stats["reads"] > 0
        assert expanded != QUERY, "every completed read was thrown away"
        assert expanded.startswith(QUERY)

    async def test_cancellation_is_what_discards_everything(self):
        """Control: without a deadline the caller's wait_for loses the lot.

        This is the pre-fix behaviour and the reason the deadline exists. If
        this control ever stops failing, the tests above prove nothing.
        """
        store = _SlowGraphStore(read_seconds=0.01)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                expand_query_from_graph(QUERY, store, "default"),
                timeout=0.05,
            )
        assert store.calls, "fixture issued no reads at all"

    async def test_no_deadline_keeps_the_old_unbounded_behaviour(self):
        store = _SlowGraphStore(read_seconds=0.0)
        stats: dict[str, float] = {}
        expanded = await expand_query_from_graph(QUERY, store, "default", stats_out=stats)
        assert stats["truncated"] == 0.0
        assert expanded != QUERY


class TestFanOutIsBounded:
    async def test_lookups_actually_overlap(self):
        store = _SlowGraphStore(read_seconds=0.002)
        await expand_query_from_graph(QUERY, store, "default")
        assert store.peak_inflight > 1, "per-term lookups are still serial"

    async def test_concurrency_never_exceeds_the_bound(self):
        """Driven past the bound on purpose.

        At the shipped ``max_entities=5`` the structure caps overlap at 5, so
        asserting ``<= 8`` there would be a test that cannot fail. Twenty terms
        makes the semaphore the only thing holding the line.
        """
        many = " ".join(f"about Term{i}" for i in range(20))
        store = _SlowGraphStore(read_seconds=0.002)
        await expand_query_from_graph("what " + many, store, "default", max_entities=20)
        assert store.peak_inflight > 1
        assert store.peak_inflight <= _EXPANSION_FANOUT, store.peak_inflight

    async def test_the_duplicate_relationship_read_is_gone(self):
        """include_relationships and include_neighbors shared one call.

        They used to issue the IDENTICAL ``get_relationships(eid)`` twice per
        entity -- up to ten duplicated native reads per recall.
        """
        store = _SlowGraphStore(read_seconds=0.0)
        await expand_query_from_graph(QUERY, store, "default")
        rel_calls = [c for c in store.calls if c.startswith("rels:")]
        assert len(rel_calls) == len(set(rel_calls)), rel_calls

    async def test_relationship_reads_are_still_made(self):
        """Control: the dedupe must not have deleted the read entirely."""
        store = _SlowGraphStore(read_seconds=0.0)
        await expand_query_from_graph(QUERY, store, "default")
        assert any(c.startswith("rels:") for c in store.calls)
        assert any(c.startswith("entity:") for c in store.calls)


class TestExpansionOutputIsUnchanged:
    async def test_expansion_text_matches_the_serial_reference(self):
        """Parallel lookups must not reorder the expansion.

        Reference implementation = the pre-fix serial walk, written out here so
        the assertion is against behaviour rather than against a golden string
        someone can silently re-bless.
        """
        store = _SlowGraphStore(read_seconds=0.0)
        expanded = await expand_query_from_graph(QUERY, store, "default")

        ref_store = _SlowGraphStore(read_seconds=0.0)
        parts: list[str] = []
        seen: set[str] = set()
        from engram.retrieval.graph_expansion import _extract_query_terms

        for term in _extract_query_terms(QUERY)[:5]:
            candidates = await ref_store.find_entity_candidates(term, "default")
            for entity in candidates[:2]:
                if entity.id in seen:
                    continue
                seen.add(entity.id)
                if entity.name:
                    parts.append(entity.name)
                if entity.summary and len(entity.summary) > 5:
                    parts.append(entity.summary)
                rels = await ref_store.get_relationships(entity.id, group_id="default")
                for rel in rels[:5]:
                    if rel.predicate:
                        other = (
                            rel.target_id if rel.source_id == entity.id else rel.source_id
                        )
                        parts.append(f"{rel.predicate} {other}")
                for rel in rels[:3]:
                    other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
                    if other_id and other_id != entity.id:
                        neighbor = await ref_store.get_entity(other_id, "default")
                        if neighbor and neighbor.name:
                            parts.append(neighbor.name)

        expansion = " ".join(parts)[:500]
        assert expanded == f"{QUERY} {expansion}"

    async def test_a_query_with_no_terms_reads_nothing(self):
        """The live common case: an all-lowercase question touches no store."""
        store = _SlowGraphStore(read_seconds=0.05)
        stats: dict[str, float] = {}
        expanded = await expand_query_from_graph(
            "how do consolidation phases get scheduled",
            store,
            "default",
            deadline_seconds=0.075,
            stats_out=stats,
        )
        assert store.calls == []
        assert stats["reads"] == 0.0
        assert expanded == "how do consolidation phases get scheduled"
