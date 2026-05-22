from __future__ import annotations

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
