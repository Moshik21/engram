"""The packet cache must not serve arm A's packets to arm B (ticket #29 / AUDIT-14).

WHY THIS FILE EXISTS. Until 2026-07-24 the key was

    f"{group_id}:{scope}:{topic_digest}:{project_digest}"

with a 300 s TTL and SQLite persistence that survives a restart. Nothing in it
named the build, the config, or the arm. So in an A/B whose arms differ by
config or by code, arm B could be served arm A's cached packets for the same
query, and the run would report "no difference" for a change of any size — with
a clean, low-variance, entirely convincing number. Restarting between arms did
not clear it.

Measured, not theoretical: the first ``engram meter`` capture reported sd = 0.0
and "N >= 2 runs/arm", both false, because ``cache_satisfied`` had served
24/168 probes.

Every test below is written so that reverting the fix turns it RED. The arms
used are real ledger arms — the spreading knobs from tickets #32/#33 and
``recall_profile`` — not invented ones, because a fixture assembled from
imagination inherits the author's blind spots (STANDING_GOAL §2.7).
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from engram.config import ActivationConfig
from engram.retrieval.packet_cache import (
    FINGERPRINT_EXCLUDED_FIELDS,
    KEY_SCHEMA,
    NAMESPACE_ENV_VAR,
    MemoryPacketCache,
    PacketCacheIdentity,
    packet_cache_identity,
)

# Two arms that MUST produce different answers. Arm B is exactly the
# configuration ticket #33 calls out as reproducing pre-fix spreading
# behaviour: ``max_reads=0 + budget=0`` means the traversal never runs.
ARM_A = {"retrieval_spread_max_reads": 64, "retrieval_spread_traversal_budget_ms": 50}
ARM_B = {"retrieval_spread_max_reads": 0, "retrieval_spread_traversal_budget_ms": 0}

PACKET = [{"title": "Engram", "summary": "arm A packet", "entity_ids": ["ent_1"]}]


def _legacy_build_key(
    group_id: str,
    scope: str,
    topic_digest: str = "8e35c2cd3bf6641b",
    project_digest: str = "none",
) -> str:
    """The pre-fix key, verbatim (``packet_cache.py:117`` before this fix)."""
    return f"{group_id}:{scope}:{topic_digest}:{project_digest}"


def _cache(overrides: dict, path=None, namespace: str | None = None) -> MemoryPacketCache:
    cfg = ActivationConfig(**overrides)
    return MemoryPacketCache(
        default_ttl_seconds=300.0,
        persistence_path=path,
        identity=packet_cache_identity(cfg, runtime_mode="lite", namespace=namespace),
    )


class TestTheKeyCannotCollideAcrossArms:
    def test_arms_collided_on_the_legacy_key_and_do_not_now(self) -> None:
        """The trap, then the fix, in one assertion pair.

        The first two asserts are the bug: the old key is byte-identical for two
        configurations that cannot return the same rows. The third is the fix.
        Reverting ``build_key`` makes the third fail.
        """
        legacy_a = _legacy_build_key("default", "explicit_recall:mcp_recall")
        legacy_b = _legacy_build_key("default", "explicit_recall:mcp_recall")
        assert legacy_a == legacy_b, "pre-fix keys were identical across arms — that was the bug"

        arm_a = _cache(ARM_A)
        arm_b = _cache(ARM_B)
        key_a = arm_a.build_key(
            group_id="default",
            scope="explicit_recall:mcp_recall",
            topic_hint="what is the recall budget",
        )
        key_b = arm_b.build_key(
            group_id="default",
            scope="explicit_recall:mcp_recall",
            topic_hint="what is the recall budget",
        )
        assert key_a != key_b
        assert key_a.startswith(f"{KEY_SCHEMA}:")

    # NOTE: there is deliberately no "two in-process caches" version of the
    # test below. A first draft had one and it passed with the fix reverted —
    # two MemoryPacketCache objects hold separate ``_entries`` maps, so they
    # cannot share regardless of the key. Cross-arm sharing happens through the
    # SQLite sidecar, which is what TestPersistenceDoesNotLeakAcrossArms
    # exercises, and that suite does go red when the key is reverted.

    def test_identical_config_still_shares_the_cache(self) -> None:
        """The fix must not degenerate into "never cache anything"."""
        first = _cache(ARM_A)
        second = _cache(ARM_A)
        assert first.identity.fingerprint == second.identity.fingerprint
        first.put(
            group_id="default",
            scope="identity_core",
            packets=PACKET,
            now=10.0,
        )
        assert second.build_key(group_id="default", scope="identity_core") == first.build_key(
            group_id="default", scope="identity_core"
        )


class TestPersistenceDoesNotLeakAcrossArms:
    """The sidecar is the half that survives a restart, so it is the half that
    matters for an A/B that restarts the server between arms.

    These use wall-clock timestamps deliberately: the persistence layer expires
    rows against ``time.time()``, so an injected synthetic ``now`` silently
    deletes every row before the assertion runs. The first draft of this file
    did exactly that and its positive control caught it.
    """

    def test_arm_b_does_not_load_arm_a_rows_from_the_sidecar(self, tmp_path) -> None:
        now = time.time()
        sidecar = tmp_path / "packet-cache.sqlite3"
        arm_a = _cache(ARM_A, path=sidecar)
        arm_a.put(
            group_id="default",
            scope="explicit_recall:mcp_recall",
            topic_hint="q",
            packets=PACKET,
            now=now,
        )
        arm_b = _cache(ARM_B, path=sidecar)  # a "restart" into the other arm
        assert (
            arm_b.get(
                group_id="default",
                scope="explicit_recall:mcp_recall",
                topic_hint="q",
                now=now + 10,
            )
            is None
        )
        # Positive control: the same restart in the SAME arm still warm-starts,
        # so the miss above is isolation and not a broken sidecar.
        arm_a_restarted = _cache(ARM_A, path=sidecar)
        assert (
            arm_a_restarted.get(
                group_id="default",
                scope="explicit_recall:mcp_recall",
                topic_hint="q",
                now=now + 10,
            )
            is not None
        )

    def test_arm_a_packets_are_not_served_through_the_degraded_lane(self, tmp_path) -> None:
        """``recent_packets`` reads the in-memory map WITHOUT rebuilding a key.

        This is the leak a key-only fix would have missed: the degraded-fallback
        lane (``get_recent_cached_memory_packets``) never calls ``build_key``,
        so it would happily hand arm B a packet arm A built.
        """
        now = time.time()
        sidecar = tmp_path / "packet-cache.sqlite3"
        arm_a = _cache(ARM_A, path=sidecar)
        arm_a.put(
            group_id="default",
            scope="explicit_recall:mcp_recall",
            topic_hint="q",
            packets=PACKET,
            now=now,
        )
        assert arm_a.recent_packets(group_id="default", now=now + 10)

        arm_b = _cache(ARM_B, path=sidecar)
        assert arm_b.recent_packets(group_id="default", now=now + 10) == []
        # And it is visible, not silent: the entries are counted as foreign.
        assert arm_b.summary(group_id="default", now=now + 10)["foreign_entry_count"] == 1

    def test_pre_fix_rows_are_purged_not_resurrected(self, tmp_path) -> None:
        """A sidecar written before this fix must not come back to life.

        Its keys can no longer be produced by ``build_key``, so ``get`` cannot
        reach them — but ``recent_packets`` could, so they are deleted on load.
        """
        now = time.time()
        sidecar = tmp_path / "packet-cache.sqlite3"
        _cache(ARM_A, path=sidecar)  # creates the schema
        legacy_key = _legacy_build_key("default", "explicit_recall:mcp_recall")
        with sqlite3.connect(sidecar) as db:
            db.execute(
                "INSERT INTO memory_packet_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    legacy_key,
                    "default",
                    "explicit_recall:mcp_recall",
                    "q",
                    None,
                    '[{"title": "stale pre-fix packet"}]',
                    "[]",
                    "[]",
                    "[]",
                    10.0,
                    10.0,
                    1e12,  # far-future expiry: it would otherwise be perfectly fresh
                    None,
                    0.0,
                    0,
                    None,
                ),
            )
            db.commit()

        cache = _cache(ARM_A, path=sidecar)
        assert cache.recent_packets(group_id="default", now=now + 10) == []
        assert (
            cache.get(
                group_id="default",
                scope="explicit_recall:mcp_recall",
                topic_hint="q",
                now=now + 10,
            )
            is None
        )
        with sqlite3.connect(sidecar) as db:
            remaining = db.execute(
                "SELECT COUNT(*) FROM memory_packet_cache WHERE cache_key = ?",
                (legacy_key,),
            ).fetchone()[0]
        assert remaining == 0, "a pre-fix row survived the migration and can still be served"


class TestWhatTheFingerprintCovers:
    """The design call: over-inclusion costs at most one TTL window of warmth;
    under-inclusion costs a clean, low-variance, wrong null result."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"recall_profile": "wave2"},
            {"spread_candidate_injection_max": 450},  # ticket #32's harness divergence
            {"pool_total_limit": 200},  # ticket #28's recall-depth lever
            {"reranker_enabled": False},
            {"retrieval_spread_max_reads": 0},
        ],
    )
    def test_a_knob_that_changes_the_answer_changes_the_fingerprint(self, overrides) -> None:
        base = packet_cache_identity(ActivationConfig(), runtime_mode="lite")
        arm = packet_cache_identity(ActivationConfig(**overrides), runtime_mode="lite")
        assert base.fingerprint != arm.fingerprint

    @pytest.mark.parametrize(
        "overrides",
        [
            {"recall_packet_cache_ttl_seconds": 17.0},
            {"recall_packet_cache_max_entries": 16},
            {"recall_packet_cache_enabled": False},
            {"recall_packet_cache_persistence_enabled": False},
            {"recall_packet_cache_path": "/tmp/elsewhere.sqlite3"},
        ],
    )
    def test_cache_plumbing_does_not_evict(self, overrides) -> None:
        """Retuning the cache itself must not throw the cache away.

        A fingerprint over the whole config would evict every entry when a rig
        changes the TTL — the one change that provably cannot alter an answer.
        """
        base = packet_cache_identity(ActivationConfig(), runtime_mode="lite")
        tuned = packet_cache_identity(ActivationConfig(**overrides), runtime_mode="lite")
        assert base.fingerprint == tuned.fingerprint

    def test_the_exclusion_list_is_only_cache_plumbing(self) -> None:
        """Guards the escape hatch: excluding a real retrieval knob would
        silently reopen the trap, so every excluded name must be cache
        plumbing and must carry a reason."""
        assert set(FINGERPRINT_EXCLUDED_FIELDS) == {
            "recall_packet_cache_enabled",
            "recall_packet_cache_ttl_seconds",
            "recall_packet_cache_max_entries",
            "recall_packet_cache_persistence_enabled",
            "recall_packet_cache_path",
        }
        assert all(reason.strip() for reason in FINGERPRINT_EXCLUDED_FIELDS.values())

    def test_runtime_mode_is_part_of_the_identity(self) -> None:
        cfg = ActivationConfig()
        assert (
            packet_cache_identity(cfg, runtime_mode="lite").fingerprint
            != packet_cache_identity(cfg, runtime_mode="helix").fingerprint
        )

    def test_build_component_is_present_and_labelled(self) -> None:
        identity = packet_cache_identity(ActivationConfig(), runtime_mode="lite")
        payload = identity.to_dict()
        assert payload["build"], "no build component: two versions would share packets"
        # Absent rather than plausible when the source cannot be read.
        assert (payload["sourceDigest"] is None) == (payload["sourceDigestStatus"] == "unavailable")
        assert payload["sourceDigestScope"]

    def test_namespace_isolates_arms_that_differ_by_neither_config_nor_code(
        self, monkeypatch
    ) -> None:
        """The explicit measurement lever: two arms over different planted
        corpora have identical config and identical code."""
        cfg = ActivationConfig()
        base = packet_cache_identity(cfg, runtime_mode="lite")
        arm = packet_cache_identity(cfg, runtime_mode="lite", namespace="arm-b")
        assert base.fingerprint != arm.fingerprint

        monkeypatch.setenv(NAMESPACE_ENV_VAR, "arm-c")
        from_env = packet_cache_identity(cfg, runtime_mode="lite")
        assert from_env.fingerprint not in {base.fingerprint, arm.fingerprint}
        assert from_env.namespace == "arm-c"

    def test_an_unfingerprinted_cache_says_so(self) -> None:
        """Absence must be legible. A cache built with no identity must not
        report a plausible-looking digest."""
        cache = MemoryPacketCache()
        summary = cache.summary(group_id="default", now=1.0)
        assert summary["fingerprint"] == "unfingerprinted"
        assert summary["identity"]["status"] == "unfingerprinted"


class TestTheProductionCacheIsActuallyFingerprinted:
    """The consumer test. The identity could be built perfectly and never
    passed to the cache the server actually uses — this project's dominant bug
    class. Neutering the ``identity=`` argument in ``graph_manager`` turns this
    red while every test above stays green."""

    def test_graph_manager_wires_a_real_identity(self) -> None:
        from unittest.mock import MagicMock

        from engram.graph_manager import GraphManager

        def build(**overrides) -> dict:
            cfg = ActivationConfig(recall_packet_cache_persistence_enabled=False, **overrides)
            manager = GraphManager(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                cfg=cfg,
                runtime_mode="lite",
            )
            return manager.get_memory_packet_cache_summary("default")

        arm_a = build(**ARM_A)
        arm_b = build(**ARM_B)
        assert arm_a["identity"]["status"] == "fingerprinted"
        assert arm_a["fingerprint"] != arm_b["fingerprint"]
        # And the provenance a measurement rig needs is on the summary that
        # feeds /api/knowledge/runtime.
        assert arm_a["ttl_seconds"] == 300.0
        assert arm_a["enabled"] is True
        assert build(recall_packet_cache_enabled=False)["enabled"] is False


class TestIdentityDataclass:
    def test_key_prefix_is_schema_and_fingerprint(self) -> None:
        identity = PacketCacheIdentity(fingerprint="abc123")
        assert identity.key_prefix == f"{KEY_SCHEMA}:abc123"
