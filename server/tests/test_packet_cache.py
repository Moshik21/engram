from __future__ import annotations

import sqlite3
import time

from engram.retrieval.packet_cache import MemoryPacketCache, packet_cache_json_size


def test_packet_cache_returns_fresh_hits_and_counts_usage() -> None:
    cache = MemoryPacketCache(default_ttl_seconds=60)
    cache.put(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="Engram AXI",
        packets=[
            {
                "title": "Engram",
                "entity_ids": ["ent_1"],
                "episode_ids": ["ep_1"],
                "relationship_ids": ["rel_1"],
            }
        ],
        build_duration_ms=12.5,
        now=10.0,
    )

    hit = cache.get(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="Engram AXI",
        now=20.0,
    )

    assert hit is not None
    assert hit.packets[0]["title"] == "Engram"
    assert hit.entry.hit_count == 1
    assert hit.entry.last_hit_at == 20.0
    assert cache.summary(group_id="default", now=20.0)["fresh_count"] == 1


def test_packet_cache_returns_recent_fresh_packets_by_scope() -> None:
    cache = MemoryPacketCache(default_ttl_seconds=60)
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[{"title": "Engram project", "entity_ids": ["ent_1"]}],
        now=10.0,
    )
    cache.put(
        group_id="default",
        scope="identity_core",
        packets=[{"title": "User preference", "entity_ids": ["ent_2"]}],
        now=20.0,
    )
    cache.put(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="other",
        packets=[{"title": "Other", "entity_ids": ["ent_3"]}],
        now=30.0,
    )

    packets = cache.recent_packets(
        group_id="default",
        scopes=("identity_core", "project_home"),
        limit_packets=2,
        now=25.0,
    )

    assert [packet["title"] for packet in packets] == [
        "User preference",
        "Engram project",
    ]
    assert [packet["_cache_scope"] for packet in packets] == [
        "identity_core",
        "project_home",
    ]


def test_packet_cache_recent_packets_deduplicates_shared_packet_payloads() -> None:
    cache = MemoryPacketCache(default_ttl_seconds=60)
    packet = {
        "title": "Project File: docs/install/helix.md",
        "summary": "Helix install docs",
        "provenance": ["file:docs/install/helix.md"],
    }
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[packet],
        now=10.0,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="native PyO3",
        packets=[packet],
        now=20.0,
    )

    packets = cache.recent_packets(
        group_id="default",
        scopes=("project_home",),
        limit_packets=5,
        now=25.0,
    )

    assert packets == [
        {
            **packet,
            "_cache_scope": "project_home",
        },
    ]


def test_packet_cache_expires_and_invalidates_by_source_ids() -> None:
    cache = MemoryPacketCache(default_ttl_seconds=5)
    cache.put(
        group_id="default",
        scope="auto_recall_packet",
        topic_hint="Phoenix",
        packets=[{"title": "Phoenix", "entityIds": ["ent_project"]}],
        now=10.0,
    )
    cache.put(
        group_id="default",
        scope="auto_recall_packet",
        topic_hint="Mesa",
        packets=[{"title": "Mesa", "entityIds": ["ent_other"]}],
        now=10.0,
    )

    assert cache.get(
        group_id="default",
        scope="auto_recall_packet",
        topic_hint="Phoenix",
        now=14.0,
    )
    assert cache.get(
        group_id="default",
        scope="auto_recall_packet",
        topic_hint="Phoenix",
        now=16.0,
    ) is None
    invalidated = cache.invalidate(
        group_id="default",
        entity_ids=["ent_other"],
        now=17.0,
    )

    assert invalidated == 1
    assert cache.get(
        group_id="default",
        scope="auto_recall_packet",
        topic_hint="Mesa",
        now=18.0,
    ) is None


def test_packet_cache_clear_and_json_size_are_deterministic() -> None:
    cache = MemoryPacketCache(default_ttl_seconds=60)
    packets = [{"b": 2, "a": 1}]
    cache.put(group_id="default", scope="s", packets=packets, now=1.0)
    cache.put(group_id="other", scope="s", packets=packets, now=1.0)

    assert packet_cache_json_size(packets) == len('[{"a":1,"b":2}]')
    assert cache.clear(group_id="default") == 1
    assert cache.summary(group_id="default")["entry_count"] == 0
    assert cache.summary(group_id="other")["entry_count"] == 1


def test_packet_cache_persists_entries_across_instances(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    packets = [
        {
            "title": "Engram native path",
            "entity_ids": ["ent_native"],
            "episode_ids": ["ep_native"],
        }
    ]

    cache = MemoryPacketCache(
        default_ttl_seconds=0,
        persistence_path=path,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        project_path="/Users/konnermoshier/Engram",
        packets=packets,
        now=10.0,
    )

    reopened = MemoryPacketCache(
        default_ttl_seconds=0,
        persistence_path=path,
    )
    hit = reopened.get(
        group_id="default",
        scope="project_home",
        project_path="/Users/konnermoshier/Engram",
        now=20.0,
    )

    assert hit is not None
    assert hit.packets[0]["title"] == "Engram native path"
    assert hit.entry.source_entity_ids == {"ent_native"}
    assert reopened.summary(group_id="default")["persistent"] is True
    assert reopened.summary(group_id="default")["path"] == str(path)


def test_packet_cache_eviction_keeps_startup_resident_project_packets(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    cache = MemoryPacketCache(
        max_entries=2,
        default_ttl_seconds=0,
        persistence_path=path,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[{"title": "Stable Engram project packet"}],
        now=10.0,
    )
    cache.put(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="first explicit",
        packets=[{"title": "First explicit packet"}],
        now=20.0,
    )
    cache.put(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="second explicit",
        packets=[{"title": "Second explicit packet"}],
        now=30.0,
    )

    reopened = MemoryPacketCache(
        max_entries=2,
        default_ttl_seconds=0,
        persistence_path=path,
    )

    assert (
        reopened.get(
            group_id="default",
            scope="project_home",
            topic_hint="Engram",
            sync_persistent=False,
            now=40.0,
        )
        is not None
    )
    assert (
        reopened.get(
            group_id="default",
            scope="explicit_recall:mcp_recall",
            topic_hint="first explicit",
            now=40.0,
        )
        is None
    )
    latest = reopened.get(
        group_id="default",
        scope="explicit_recall:mcp_recall",
        topic_hint="second explicit",
        sync_persistent=False,
        now=40.0,
    )
    assert latest is not None
    assert latest.packets[0]["title"] == "Second explicit packet"


def test_packet_cache_local_only_entries_skip_persistence(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    cache = MemoryPacketCache(
        default_ttl_seconds=60,
        persistence_path=path,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram dogfood",
        project_path="/Users/konnermoshier/Engram",
        packets=[{"title": "Local project-file fallback packet"}],
        persist=False,
        now=10.0,
    )

    hit = cache.get(
        group_id="default",
        scope="project_home",
        topic_hint="Engram dogfood",
        project_path="/Users/konnermoshier/Engram",
        now=20.0,
    )
    reopened = MemoryPacketCache(
        default_ttl_seconds=60,
        persistence_path=path,
    )

    assert hit is not None
    assert hit.packets[0]["title"] == "Local project-file fallback packet"
    assert cache.summary(group_id="default", now=20.0)["fresh_count"] == 1
    assert (
        reopened.get(
            group_id="default",
            scope="project_home",
            topic_hint="Engram dogfood",
            project_path="/Users/konnermoshier/Engram",
            now=20.0,
        )
        is None
    )


def test_packet_cache_get_loads_persistent_entry_written_by_another_instance(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    reader = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)
    writer = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)

    writer.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        project_path="/Users/konnermoshier/Engram",
        packets=[{"title": "Shared project packet"}],
        now=10.0,
    )

    hit = reader.get(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        project_path="/Users/konnermoshier/Engram",
        now=20.0,
    )

    assert hit is not None
    assert hit.packets[0]["title"] == "Shared project packet"
    assert reader.summary(group_id="default")["hit_count"] == 1


def test_packet_cache_recent_packets_syncs_persistent_entries(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    reader = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)
    writer = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)

    writer.put(
        group_id="default",
        scope="project_home",
        packets=[{"title": "Recent shared packet"}],
        now=10.0,
    )

    packets = reader.recent_packets(
        group_id="default",
        scopes=("project_home",),
        now=20.0,
    )

    assert packets == [{"title": "Recent shared packet", "_cache_scope": "project_home"}]


def test_packet_cache_recent_packets_can_skip_persistent_sync(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    reader = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)
    writer = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)

    writer.put(
        group_id="default",
        scope="project_home",
        packets=[{"title": "Recent shared packet"}],
        now=10.0,
    )

    assert (
        reader.recent_packets(
            group_id="default",
            scopes=("project_home",),
            sync_persistent=False,
            now=20.0,
        )
        == []
    )


def test_packet_cache_get_respects_persistent_clear_from_other_instance(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    reader = MemoryPacketCache(
        default_ttl_seconds=0,
        persistence_path=path,
        persistent_sync_interval_seconds=0,
    )
    writer = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)

    writer.put(
        group_id="default",
        scope="project_home",
        packets=[{"title": "Cleared shared packet"}],
        now=10.0,
    )
    assert reader.get(group_id="default", scope="project_home", now=20.0) is not None

    writer.clear(group_id="default")

    assert reader.get(group_id="default", scope="project_home", now=30.0) is None


def test_packet_cache_persistent_invalidation_survives_reopen(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    cache = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)
    cache.put(
        group_id="default",
        scope="project_home",
        packets=[{"title": "Old fact", "entity_ids": ["ent_old"]}],
        now=10.0,
    )
    assert cache.invalidate(group_id="default", entity_ids=["ent_old"], now=15.0) == 1

    reopened = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)

    assert reopened.summary(group_id="default")["invalidated_count"] == 1
    assert reopened.get(group_id="default", scope="project_home", now=20.0) is None


def test_packet_cache_locked_persistence_miss_returns_quickly(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    writer = MemoryPacketCache(default_ttl_seconds=0, persistence_path=path)
    writer.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[{"title": "Shared packet"}],
        now=10.0,
    )
    lock = sqlite3.connect(path)
    try:
        lock.execute("BEGIN EXCLUSIVE")
        reader = MemoryPacketCache(
            default_ttl_seconds=0,
            persistence_path=path,
            persistence_timeout_seconds=0.01,
        )

        started = time.perf_counter()
        hit = reader.get(
            group_id="default",
            scope="project_home",
            topic_hint="missing topic",
            now=20.0,
        )
        elapsed = time.perf_counter() - started
    finally:
        lock.rollback()
        lock.close()

    assert hit is None
    assert elapsed < 0.25
    assert reader.get(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        now=21.0,
    )


def test_packet_cache_locked_persistence_keeps_in_memory_hit(tmp_path) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    cache = MemoryPacketCache(
        default_ttl_seconds=60,
        persistence_path=path,
        persistence_timeout_seconds=0.01,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[{"title": "Hot packet"}],
        now=10.0,
    )
    lock = sqlite3.connect(path)
    try:
        lock.execute("BEGIN EXCLUSIVE")

        started = time.perf_counter()
        hit = cache.get(
            group_id="default",
            scope="project_home",
            topic_hint="Engram",
            now=20.0,
        )
        elapsed = time.perf_counter() - started
    finally:
        lock.rollback()
        lock.close()

    assert hit is not None
    assert hit.packets[0]["title"] == "Hot packet"
    assert elapsed < 0.25


def test_packet_cache_hot_persistent_hit_skips_sqlite_until_sync_interval(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "packet-cache.sqlite3"
    cache = MemoryPacketCache(
        default_ttl_seconds=60,
        persistence_path=path,
        persistent_sync_interval_seconds=300,
    )
    cache.put(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        packets=[{"title": "Hot packet"}],
        now=10.0,
    )

    def fail_sync(*_args, **_kwargs):
        raise AssertionError("hot hit should not touch persistent sqlite")

    monkeypatch.setattr(cache, "_load_persistent_entry", fail_sync)
    monkeypatch.setattr(cache, "_persist_entry", fail_sync)

    hit = cache.get(
        group_id="default",
        scope="project_home",
        topic_hint="Engram",
        now=20.0,
    )

    assert hit is not None
    assert hit.packets[0]["title"] == "Hot packet"
    assert hit.entry.hit_count == 1
