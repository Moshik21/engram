"""Tests for community-aware spreading activation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from engram.activation.bfs import BFSStrategy
from engram.activation.community import CommunityStore, label_propagation
from engram.activation.ppr import PPRStrategy
from engram.activation.spreading import spread_activation
from engram.config import ActivationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_neighbor_provider(adjacency: dict[str, list[tuple[str, float, str]]]):
    """Build an async mock neighbor provider from an adjacency dict."""
    provider = AsyncMock()

    async def get_neighbors(entity_id, group_id=None):
        return adjacency.get(entity_id, [])

    provider.get_active_neighbors_with_weights = AsyncMock(side_effect=get_neighbors)
    return provider


def _two_cliques_adjacency():
    """Two 5-node cliques connected by a single bridge (a5-b1).

    Clique A: a1-a5 (fully connected)
    Clique B: b1-b5 (fully connected)
    Bridge: a5 <-> b1
    """
    adj: dict[str, list[tuple[str, float, str]]] = {}
    clique_a = [f"a{i}" for i in range(1, 6)]
    clique_b = [f"b{i}" for i in range(1, 6)]

    for node in clique_a:
        neighbors = []
        for other in clique_a:
            if other != node:
                neighbors.append((other, 1.0, "KNOWS"))
        if node == "a5":
            neighbors.append(("b1", 1.0, "KNOWS"))
        adj[node] = neighbors

    for node in clique_b:
        neighbors = []
        for other in clique_b:
            if other != node:
                neighbors.append((other, 1.0, "KNOWS"))
        if node == "b1":
            neighbors.append(("a5", 1.0, "KNOWS"))
        adj[node] = neighbors

    return adj, clique_a, clique_b


# ===========================================================================
# TestLabelPropagation
# ===========================================================================


class TestLabelPropagation:
    @pytest.mark.asyncio
    async def test_two_cliques(self):
        """Two 5-node cliques with 1 bridge -> 2 communities."""
        adj, clique_a, clique_b = _two_cliques_adjacency()
        provider = _build_neighbor_provider(adj)
        all_ids = clique_a + clique_b

        labels = await label_propagation(
            provider, "test", entity_ids=all_ids, max_iterations=10, seed=42,
        )

        # All nodes in clique A should share one label
        a_labels = {labels[n] for n in clique_a}
        assert len(a_labels) == 1, f"Clique A has multiple labels: {a_labels}"

        # All nodes in clique B should share one label
        b_labels = {labels[n] for n in clique_b}
        assert len(b_labels) == 1, f"Clique B has multiple labels: {b_labels}"

        # The two cliques should have different labels
        assert a_labels != b_labels

    @pytest.mark.asyncio
    async def test_single_component(self):
        """Fully connected graph -> 1 community."""
        nodes = ["n1", "n2", "n3", "n4"]
        adj = {}
        for node in nodes:
            adj[node] = [
                (other, 1.0, "KNOWS") for other in nodes if other != node
            ]
        provider = _build_neighbor_provider(adj)

        labels = await label_propagation(
            provider, "test", entity_ids=nodes, max_iterations=10, seed=42,
        )
        unique_labels = set(labels.values())
        assert len(unique_labels) == 1

    @pytest.mark.asyncio
    async def test_isolated_nodes(self):
        """No neighbors -> each keeps own label."""
        nodes = ["n1", "n2", "n3"]
        adj = {n: [] for n in nodes}
        provider = _build_neighbor_provider(adj)

        labels = await label_propagation(
            provider, "test", entity_ids=nodes, max_iterations=10, seed=42,
        )
        # Each node should keep its own label
        assert labels["n1"] != labels["n2"]
        assert labels["n2"] != labels["n3"]

    @pytest.mark.asyncio
    async def test_deterministic(self):
        """Same seed -> identical output."""
        adj, clique_a, clique_b = _two_cliques_adjacency()
        provider = _build_neighbor_provider(adj)
        all_ids = clique_a + clique_b

        labels1 = await label_propagation(
            provider, "test", entity_ids=all_ids, seed=42,
        )
        labels2 = await label_propagation(
            provider, "test", entity_ids=all_ids, seed=42,
        )
        assert labels1 == labels2

    @pytest.mark.asyncio
    async def test_max_iterations_1(self):
        """Limited iterations may not converge for complex graphs."""
        adj, clique_a, clique_b = _two_cliques_adjacency()
        provider = _build_neighbor_provider(adj)
        all_ids = clique_a + clique_b

        labels = await label_propagation(
            provider, "test", entity_ids=all_ids, max_iterations=1, seed=42,
        )
        # Should still produce valid labels for all nodes
        assert len(labels) == len(all_ids)
        for nid in all_ids:
            assert nid in labels

    @pytest.mark.asyncio
    async def test_four_clusters(self):
        """100 nodes, 4 clusters with sparse bridges."""
        clusters = {
            f"c{ci}": [f"c{ci}_n{i}" for i in range(25)]
            for ci in range(4)
        }
        adj: dict[str, list[tuple[str, float, str]]] = {}
        all_ids = []

        for cname, members in clusters.items():
            all_ids.extend(members)
            for node in members:
                neighbors = [
                    (other, 1.0, "KNOWS")
                    for other in members if other != node
                ]
                adj[node] = neighbors

        # Add one bridge between each adjacent pair
        bridge_pairs = [("c0", "c1"), ("c1", "c2"), ("c2", "c3")]
        for ca, cb in bridge_pairs:
            a_node = clusters[ca][0]
            b_node = clusters[cb][0]
            adj[a_node].append((b_node, 1.0, "KNOWS"))
            adj[b_node].append((a_node, 1.0, "KNOWS"))

        provider = _build_neighbor_provider(adj)
        labels = await label_propagation(
            provider, "test", entity_ids=all_ids, max_iterations=20, seed=42,
        )

        # Should detect 4 communities
        unique_labels = set(labels.values())
        assert len(unique_labels) == 4

    @pytest.mark.asyncio
    async def test_empty_entity_ids(self):
        """Empty entity_ids returns empty dict."""
        provider = _build_neighbor_provider({})
        labels = await label_propagation(
            provider, "test", entity_ids=[], seed=42,
        )
        assert labels == {}


# ===========================================================================
# TestCommunityStore
# ===========================================================================


class TestCommunityStore:
    def test_get_set(self):
        """set_assignments + get_community round-trip."""
        store = CommunityStore()
        store.set_assignments("g1", {"e1": "cluster_a", "e2": "cluster_b"})
        assert store.get_community("e1", "g1") == "cluster_a"
        assert store.get_community("e2", "g1") == "cluster_b"
        assert store.get_community("e3", "g1") is None

    def test_is_bridge(self):
        """True cross-community, False intra, None unknown."""
        store = CommunityStore()
        store.set_assignments("g1", {
            "e1": "cluster_a",
            "e2": "cluster_a",
            "e3": "cluster_b",
        })
        assert store.is_bridge_edge("e1", "e2", "g1") is False  # same cluster
        assert store.is_bridge_edge("e1", "e3", "g1") is True   # cross cluster
        assert store.is_bridge_edge("e1", "e4", "g1") is None   # unknown entity
        assert store.is_bridge_edge("e1", "e2", "g2") is None   # unknown group

    def test_staleness(self):
        """Fresh not stale, after TTL is stale."""
        store = CommunityStore(stale_seconds=0.01)
        assert store.is_stale("g1") is True

        store.set_assignments("g1", {"e1": "c1"})
        assert store.is_stale("g1") is False

        # Wait for TTL
        time.sleep(0.02)
        assert store.is_stale("g1") is True

    def test_group_isolation(self):
        """Group A assignments don't leak to group B."""
        store = CommunityStore()
        store.set_assignments("ga", {"e1": "cluster_a"})
        store.set_assignments("gb", {"e1": "cluster_b"})
        assert store.get_community("e1", "ga") == "cluster_a"
        assert store.get_community("e1", "gb") == "cluster_b"

    def test_clear(self):
        """Removes cached assignments."""
        store = CommunityStore()
        store.set_assignments("g1", {"e1": "c1"})
        store.set_assignments("g2", {"e2": "c2"})

        store.clear("g1")
        assert store.get_community("e1", "g1") is None
        assert store.get_community("e2", "g2") == "c2"

        store.clear()
        assert store.get_community("e2", "g2") is None


# ===========================================================================
# TestBFSCommunityFactor
# ===========================================================================


class TestBFSCommunityFactor:
    @pytest.mark.asyncio
    async def test_bridge_boost(self):
        """Cross-cluster neighbor gets higher energy than without community."""
        adj = {
            "seed": [("cross", 1.0, "KNOWS"), ("same", 1.0, "KNOWS")],
            "cross": [],
            "same": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "cross": "B", "same": "A",
        })

        cfg = ActivationConfig(
            community_spreading_enabled=True,
            community_bridge_boost=1.5,
            community_intra_dampen=0.7,
            spread_max_hops=1,
            spread_energy_budget=50.0,
        )
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        assert bonuses["cross"] > bonuses["same"]

    @pytest.mark.asyncio
    async def test_intra_dampen(self):
        """Same-cluster neighbor gets lower energy than bridge neighbor."""
        adj = {
            "seed": [("intra", 1.0, "KNOWS"), ("bridge", 1.0, "KNOWS")],
            "intra": [],
            "bridge": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "intra": "A", "bridge": "B",
        })

        cfg = ActivationConfig(
            community_spreading_enabled=True,
            community_bridge_boost=1.5,
            community_intra_dampen=0.7,
            spread_max_hops=1,
        )
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        assert bonuses["intra"] < bonuses["bridge"]

    @pytest.mark.asyncio
    async def test_disabled_no_effect(self):
        """community_spreading_enabled=False -> identical bonuses."""
        adj = {
            "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
            "n1": [],
            "n2": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {"seed": "A", "n1": "A", "n2": "B"})

        cfg_disabled = ActivationConfig(
            community_spreading_enabled=False,
            spread_max_hops=1,
        )
        strategy = BFSStrategy()
        bonuses_disabled, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg_disabled,
            group_id="g1", community_store=store,
        )
        # Both neighbors should get equal energy when disabled
        assert abs(bonuses_disabled["n1"] - bonuses_disabled["n2"]) < 1e-9

    @pytest.mark.asyncio
    async def test_no_store_no_effect(self):
        """community_store=None -> identical bonuses."""
        adj = {
            "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
            "n1": [],
            "n2": [],
        }
        provider = _build_neighbor_provider(adj)

        cfg = ActivationConfig(
            community_spreading_enabled=True,
            spread_max_hops=1,
        )
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=None,
        )
        assert abs(bonuses["n1"] - bonuses["n2"]) < 1e-9

    @pytest.mark.asyncio
    async def test_boost_dampen_ratio(self):
        """Relative ratio matches config."""
        adj = {
            "seed": [("bridge", 1.0, "KNOWS"), ("intra", 1.0, "KNOWS")],
            "bridge": [],
            "intra": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "bridge": "B", "intra": "A",
        })

        boost = 2.0
        dampen = 0.5
        cfg = ActivationConfig(
            community_spreading_enabled=True,
            community_bridge_boost=boost,
            community_intra_dampen=dampen,
            spread_max_hops=1,
        )
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        ratio = bonuses["bridge"] / bonuses["intra"]
        expected_ratio = boost / dampen
        assert abs(ratio - expected_ratio) < 1e-6


# ===========================================================================
# TestPPRCommunityFactor
# ===========================================================================


class TestPPRCommunityFactor:
    @pytest.mark.asyncio
    async def test_bridge_boost(self):
        """Cross-cluster entity gets higher PPR score."""
        adj = {
            "seed": [("cross", 1.0, "KNOWS"), ("same", 1.0, "KNOWS")],
            "cross": [],
            "same": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "cross": "B", "same": "A",
        })

        cfg = ActivationConfig(
            spreading_strategy="ppr",
            community_spreading_enabled=True,
            community_bridge_boost=1.5,
            community_intra_dampen=0.7,
            ppr_expansion_hops=1,
        )
        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        assert bonuses.get("cross", 0) > bonuses.get("same", 0)

    @pytest.mark.asyncio
    async def test_intra_dampen(self):
        """Intra-cluster gets lower PPR score."""
        adj = {
            "seed": [("intra", 1.0, "KNOWS"), ("bridge", 1.0, "KNOWS")],
            "intra": [],
            "bridge": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "intra": "A", "bridge": "B",
        })

        cfg = ActivationConfig(
            spreading_strategy="ppr",
            community_spreading_enabled=True,
            community_bridge_boost=1.5,
            community_intra_dampen=0.7,
            ppr_expansion_hops=1,
        )
        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        assert bonuses.get("intra", 0) < bonuses.get("bridge", 0)

    @pytest.mark.asyncio
    async def test_disabled_no_effect(self):
        """community_spreading_enabled=False -> equal scores."""
        adj = {
            "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
            "n1": [],
            "n2": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {"seed": "A", "n1": "A", "n2": "B"})

        cfg = ActivationConfig(
            spreading_strategy="ppr",
            community_spreading_enabled=False,
            ppr_expansion_hops=1,
        )
        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        # Equal with tolerance
        assert abs(bonuses.get("n1", 0) - bonuses.get("n2", 0)) < 1e-6


# ===========================================================================
# TestConfig
# ===========================================================================


class TestCommunityConfig:
    def test_community_config_defaults(self):
        """Default community config values."""
        cfg = ActivationConfig()
        assert cfg.community_spreading_enabled is False
        assert cfg.community_bridge_boost == 1.5
        assert cfg.community_intra_dampen == 0.7
        assert cfg.community_stale_seconds == 300.0
        assert cfg.community_max_iterations == 10

    def test_community_config_validation(self):
        """Config field constraints are enforced."""
        # bridge_boost must be >= 1.0
        with pytest.raises(ValidationError):
            ActivationConfig(community_bridge_boost=0.5)

        # bridge_boost must be <= 3.0
        with pytest.raises(ValidationError):
            ActivationConfig(community_bridge_boost=3.5)

        # intra_dampen must be >= 0.1
        with pytest.raises(ValidationError):
            ActivationConfig(community_intra_dampen=0.05)

        # intra_dampen must be <= 1.0
        with pytest.raises(ValidationError):
            ActivationConfig(community_intra_dampen=1.1)

        # stale_seconds must be >= 10.0
        with pytest.raises(ValidationError):
            ActivationConfig(community_stale_seconds=5.0)

        # max_iterations must be >= 1
        with pytest.raises(ValidationError):
            ActivationConfig(community_max_iterations=0)


# ===========================================================================
# TestIntegration
# ===========================================================================


class TestCommunityIntegration:
    @pytest.mark.asyncio
    async def test_pipeline_with_community_store(self):
        """Smoke test: spread_activation with community_store."""
        adj = {
            "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
            "n1": [],
            "n2": [],
        }
        provider = _build_neighbor_provider(adj)
        store = CommunityStore()
        store.set_assignments("g1", {
            "seed": "A", "n1": "A", "n2": "B",
        })

        cfg = ActivationConfig(
            community_spreading_enabled=True,
            community_bridge_boost=1.5,
            community_intra_dampen=0.7,
            spread_max_hops=1,
        )

        bonuses, hop_distances = await spread_activation(
            [("seed", 1.0)], provider, cfg,
            group_id="g1", community_store=store,
        )
        assert "n1" in bonuses
        assert "n2" in bonuses
        # n2 is cross-cluster -> should get more energy
        assert bonuses["n2"] > bonuses["n1"]

    def test_benchmark_method_community(self):
        """METHOD_COMMUNITY exists with correct config."""
        from engram.benchmark.methods import METHOD_COMMUNITY

        assert METHOD_COMMUNITY.name == "Community"
        assert METHOD_COMMUNITY.config.community_spreading_enabled is True
        assert METHOD_COMMUNITY.config.community_bridge_boost == 1.5
        assert METHOD_COMMUNITY.config.community_intra_dampen == 0.7
        assert METHOD_COMMUNITY.spreading_enabled is True
        assert METHOD_COMMUNITY.routing_enabled is True
