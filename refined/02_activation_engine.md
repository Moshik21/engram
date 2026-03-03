# Activation Engine -- Refined Specification

> Replaces Tech_Spec.md Section 2 ("Activation Engine (The Novel Piece)")

---

## 1. Problem Statement

The original spec has four math/design issues:

1. **Decay is too slow.** The formula `base_activation * (1 / (1 + decay_rate * log(time + 1)))` is hyperbolic, not logarithmic. The `log(t+1)` term grows so slowly that nothing ever goes dormant -- a node accessed once a month ago still retains ~70% of its activation. Human memory follows a power-law decay curve (Anderson & Schooler, 1991).

2. **Redundant signals in the retrieval scorer.** `current_activation` (0.3 weight) already encodes recency and frequency through its decay/reinforcement rules, yet the scorer also includes a separate `recency_score` (0.2) and `frequency_score` (0.1). This double-counts recency, inflating recent items at the expense of semantically strong but older matches.

3. **Spreading activation is unbounded.** The original spreading loop has no firing threshold, no normalization by node degree, no cycle protection, and no energy budget. A hub node with 200 edges would blast activation everywhere; two mutually connected nodes would amplify each other on each hop; and heavy queries could touch the entire graph.

4. **Decay timing is unspecified.** The spec mentions both "background task: periodic activation decay sweep" and real-time decay, without clarifying which is authoritative. Background sweeps are wasteful for graphs where most nodes are cold.

---

## 2. Design Principles

- **ACT-R grounding.** Use the well-studied ACT-R base-level learning equation for activation dynamics. This gives us a principled, tunable model backed by decades of cognitive science.
- **Orthogonal signals.** The retrieval scorer should combine exactly three independent signals: semantic similarity (content match), current activation (recency + frequency + associative proximity already baked in), and edge proximity (structural closeness in the graph).
- **Lazy evaluation.** Never run background decay sweeps. Compute activation on read, using stored access timestamps. This is simpler, cheaper, and exactly correct.
- **Bounded spreading.** Spreading activation must have a firing threshold, degree normalization, a visited set, and a total energy budget so it remains O(budget) regardless of graph topology.

---

## 3. Activation State Model

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ActivationState:
    node_id: str
    access_history: list[float] = field(default_factory=list)
    # Stored as Unix timestamps for fast arithmetic.
    # Capped at MAX_HISTORY_SIZE most recent entries.

    spreading_bonus: float = 0.0
    # Transient boost from spreading activation during current retrieval.
    # Reset to 0.0 at the start of each retrieval cycle.

    last_accessed: float = 0.0
    # Unix timestamp of most recent access (redundant with access_history[-1],
    # kept for O(1) staleness checks).

    access_count: int = 0
    # Lifetime access count. Kept for analytics/dashboard display.
    # NOT used in scoring -- the access_history encodes frequency directly.
```

### What changed from the original

| Original field | Disposition |
|---|---|
| `base_activation: float` | **Removed.** No longer a stored value. Base-level activation is computed lazily from `access_history`. |
| `current_activation: float` | **Removed as stored field.** Computed on read via `compute_activation()`. |
| `decay_rate: float` | **Moved to config.** A single global decay exponent `d` (default 0.5). Per-node decay rates add complexity without clear benefit at this stage. |
| `access_count: int` | Kept for dashboard display but not used in scoring. |
| `access_history: list[datetime]` | Kept. Changed to Unix float timestamps. Capped at `MAX_HISTORY_SIZE`. |

### Redis storage layout

```
# Per-node activation state
engram:{group_id}:activation:{node_id} -> Hash {
    access_history: JSON array of float timestamps,
    access_count: int,
    last_accessed: float
}

# Transient spreading bonuses (per-retrieval, use a short TTL)
engram:{group_id}:spread:{retrieval_id}:{node_id} -> float
```

Spreading bonuses are ephemeral -- they exist only for the duration of a single retrieval call and are discarded afterward. They do not need persistence.

---

## 4. Base-Level Activation (ACT-R Power-Law Decay)

### Formula

The base-level activation of node _i_ at time _t_ is:

```
B_i(t) = ln( sum_{j=1}^{n} (t - t_j)^{-d} )
```

Where:
- `t_j` = timestamp of the _j_-th access (from `access_history`)
- `d` = decay exponent (default **0.5**, range [0.1, 1.0])
- `n` = number of recorded accesses (capped at `MAX_HISTORY_SIZE`)

This is the ACT-R base-level learning equation. It naturally encodes both **recency** (recent accesses contribute large `(t - t_j)^{-d}` terms) and **frequency** (more accesses means more terms in the sum, raising the total).

### Normalization

Raw `B_i` values are unbounded (can go negative for cold nodes, or arbitrarily positive for hot ones). We normalize to [0, 1] using a sigmoid:

```
activation_i(t) = sigmoid( (B_i(t) - B_mid) / B_scale )

where sigmoid(x) = 1 / (1 + exp(-x))
```

- `B_mid` = the `B_i` value that maps to activation 0.5. Default: **-0.5** (a node accessed once ~1 hour ago).
- `B_scale` = controls steepness. Default: **1.0**.

These are tuned so that:
- A node accessed once 10 seconds ago has activation ~0.85
- A node accessed once 1 hour ago has activation ~0.50
- A node accessed once 7 days ago has activation ~0.10
- A node accessed 10 times over the past week has activation ~0.75

### Python pseudocode

```python
import math
from engram.config import ActivationConfig  # See Section 8


def compute_base_level(
    access_history: list[float],
    now: float,
    cfg: ActivationConfig,
) -> float:
    """Compute raw ACT-R base-level activation B_i(t)."""
    if not access_history:
        return -10.0  # Effectively zero after sigmoid

    total = 0.0
    for t_j in access_history:
        age = now - t_j
        if age < cfg.min_age_seconds:
            age = cfg.min_age_seconds  # Clamp to avoid division by zero
        total += age ** (-cfg.decay_exponent)

    return math.log(total) if total > 0 else -10.0


def normalize_activation(raw_B: float, cfg: ActivationConfig) -> float:
    """Map raw B_i to [0, 1] via sigmoid."""
    x = (raw_B - cfg.B_mid) / cfg.B_scale
    return 1.0 / (1.0 + math.exp(-x))


def compute_activation(
    access_history: list[float],
    now: float,
    cfg: ActivationConfig,
) -> float:
    """Full pipeline: access_history -> normalized activation in [0, 1]."""
    raw = compute_base_level(access_history, now, cfg)
    return normalize_activation(raw, cfg)
```

### Why power-law, not exponential?

Exponential decay (`e^{-lambda * t}`) forgets too aggressively -- anything older than a few hours drops to near-zero regardless of how many times it was accessed. Power-law decay (`t^{-d}`) produces the long-tail behavior observed in human memory: heavily-accessed items retain activation for weeks, while one-off mentions fade within hours. This is what lets the engine surface "you mentioned Python a lot last month" without drowning in noise.

---

## 5. Frequency Reinforcement

Frequency is **already encoded** in the base-level equation -- each access adds a term to the sum, raising `B_i`. There is no separate reinforcement step.

When a node is accessed (retrieved, mentioned in ingestion, or receives spreading activation above threshold):

```python
def record_access(state: ActivationState, now: float, cfg: ActivationConfig) -> None:
    """Record a new access, maintaining history cap."""
    state.access_history.append(now)
    state.access_count += 1
    state.last_accessed = now

    # Cap history to bound computation cost
    if len(state.access_history) > cfg.max_history_size:
        # Keep the most recent entries
        state.access_history = state.access_history[-cfg.max_history_size:]
```

### Access events that trigger `record_access`:

| Event | Records access? |
|---|---|
| Node mentioned in ingested episode | Yes |
| Node returned in retrieval results | Yes |
| Node receives spreading activation above `firing_threshold` | No (spreading is transient, not a true access) |
| User explicitly views node in dashboard | Yes |

Note: spreading activation does NOT record access. This prevents runaway reinforcement where frequently-spread-to nodes accumulate phantom importance. Only genuine user/system interactions count.

---

## 6. Spreading Activation

### Algorithm

When retrieval identifies seed nodes (via semantic search), activation spreads outward through the graph. This is the mechanism that brings in associatively related context.

```python
from collections import deque


def spread_activation(
    seed_nodes: list[tuple[str, float]],  # (node_id, initial_energy)
    graph: GraphStore,
    cfg: ActivationConfig,
) -> dict[str, float]:
    """
    Spread activation from seed nodes through the graph.

    Returns a dict of {node_id: spreading_bonus} for all reached nodes.
    """
    bonuses: dict[str, float] = {}
    visited: set[str] = set()
    energy_spent: float = 0.0

    # BFS queue: (node_id, energy_to_spread, current_hop)
    queue: deque[tuple[str, float, int]] = deque()

    for node_id, energy in seed_nodes:
        queue.append((node_id, energy, 0))
        visited.add(node_id)

    while queue and energy_spent < cfg.spread_energy_budget:
        node_id, energy, hop = queue.popleft()

        if hop >= cfg.spread_max_hops:
            continue

        neighbors = graph.get_active_neighbors_with_weights(node_id, now=time.time())
        # get_active_neighbors_with_weights filters out edges where
        # valid_to is not None and valid_to < now. Expired relationships
        # (e.g., "works_at Company A" after the user changed jobs) must
        # not carry spreading activation.
        out_degree = len(neighbors)

        if out_degree == 0:
            continue

        # Degree normalization: divide by sqrt(out_degree) to prevent
        # hub nodes from dominating. sqrt (not linear) so hubs still
        # spread more than leaf nodes, just not proportionally.
        degree_factor = 1.0 / math.sqrt(out_degree)

        for neighbor_id, edge_weight in neighbors:
            spread_amount = energy * edge_weight * degree_factor * cfg.spread_decay_per_hop

            # Firing threshold: ignore tiny activations
            if spread_amount < cfg.spread_firing_threshold:
                continue

            # Energy budget check
            energy_spent += spread_amount
            if energy_spent > cfg.spread_energy_budget:
                break

            # Accumulate bonus (a node can receive spread from multiple paths)
            bonuses[neighbor_id] = bonuses.get(neighbor_id, 0.0) + spread_amount

            # Only enqueue if not visited (cycle protection)
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, spread_amount, hop + 1))

    return bonuses
```

### Key safeguards

| Safeguard | Parameter | Default | Purpose |
|---|---|---|---|
| **Firing threshold** | `spread_firing_threshold` | 0.05 | Prevents negligible activations from propagating. Keeps the frontier small. |
| **Degree normalization** | `sqrt(out_degree)` | n/a | Hub nodes (e.g., "User" with 200 edges) don't blast activation everywhere. Energy is divided by `sqrt(degree)`. |
| **Visited set** | `visited: set` | n/a | Each node is enqueued at most once. Prevents cycles (A -> B -> A) from amplifying activation. |
| **Energy budget** | `spread_energy_budget` | 5.0 | Total energy that can be distributed in one retrieval. Once exhausted, spreading stops. Bounds worst-case cost. |
| **Max hops** | `spread_max_hops` | 2 | Hard depth limit. Even with budget remaining, don't go deeper than N hops. |
| **Per-hop decay** | `spread_decay_per_hop` | 0.5 | Each hop halves the energy. Combined with firing threshold, this naturally limits reach. |
| **Temporal edge filter** | `valid_to` on edges | n/a | Edges with `valid_to < now` are excluded from neighbor traversal. Expired relationships do not carry activation. |

### Why sqrt(out_degree)?

Linear normalization (`1/out_degree`) would make hub nodes useless for spreading -- a node with 200 edges would spread 1/200th of its energy per edge, which falls below the firing threshold immediately. No normalization makes hubs dominate everything. `sqrt` is the standard middle ground used in GNN message-passing (GraphSAGE, etc.) and balances hub influence vs. leaf precision.

---

## 7. Retrieval Scoring

### Revised composite score

The original spec used four signals with redundancy. The refined scorer uses three orthogonal signals:

```
score_i = w_sem * semantic_similarity_i
        + w_act * activation_i
        + w_edge * edge_proximity_i
```

| Signal | Weight | Range | What it captures |
|---|---|---|---|
| `semantic_similarity` | **0.50** | [0, 1] | Content relevance to the query (cosine similarity of embeddings). |
| `activation` | **0.35** | [0, 1] | Recency + frequency + associative priming (computed via ACT-R formula + spreading bonus). |
| `edge_proximity` | **0.15** | [0, 1] | Structural closeness in the graph to seed nodes. Captures "related by graph structure" independently of activation level. |

### Why these three are orthogonal

- **Semantic similarity** is purely about content -- "does this node's text match the query?"
- **Activation** is purely about behavioral history -- "has this node been accessed recently/frequently, and is it associatively primed right now?"
- **Edge proximity** is purely about graph structure -- "is this node close to the query's seed nodes in the graph, regardless of its activation history?"

A node can be semantically irrelevant but structurally close (your coworker's name when you ask about a project). A node can be semantically relevant but cold (a topic you discussed once months ago). A node can be highly activated but semantically off-topic (something you discussed this morning about a different subject). The three signals cover the full space.

### Why recency_score and frequency_score were removed

The original scorer had:
- `current_activation * 0.3` -- which encodes recency + frequency through the decay/reinforcement model
- `recency_score * 0.2` -- `1 / (1 + hours_since_access)`, purely recency
- `frequency_score * 0.1` -- `min(1.0, access_count / 50)`, purely frequency

This double-counts both signals. The ACT-R base-level equation already produces high activation for recently and frequently accessed nodes. Adding separate recency and frequency terms on top means recent items get an outsized advantage, drowning out semantically strong but older results. Collapsing them into a single `activation` signal with weight 0.35 fixes this.

### Computing edge_proximity

```python
def compute_edge_proximity(
    node_id: str,
    seed_node_ids: set[str],
    graph: GraphStore,
    max_hops: int = 3,
) -> float:
    """
    Compute structural proximity of node_id to the seed nodes.
    Returns 1.0 for seed nodes, decays by 0.5 per hop, 0.0 if unreachable.
    """
    if node_id in seed_node_ids:
        return 1.0

    # BFS from seed nodes (already computed during spreading; reuse the hop count)
    # This is a simplified version -- in practice, cache hop distances from spread_activation.
    min_hops = shortest_path_to_any_seed(node_id, seed_node_ids, graph, max_hops)

    if min_hops is None:
        return 0.0

    return 0.5 ** min_hops  # 1 hop = 0.5, 2 hops = 0.25, 3 hops = 0.125
```

In practice, hop distances are captured as a side effect of the spreading activation BFS, so no additional graph traversal is needed.

### Full retrieval flow (revised)

```
Query arrives
    |
    v
Embed query -> vector search -> top-K candidate node IDs (semantic_similarity scores)
    |
    v
For each candidate, compute_activation(access_history, now, cfg) -> base activation
    |
    v
Identify seed nodes = candidates with semantic_similarity >= seed_threshold (default 0.3)
    |
    v
spread_activation(seeds, graph, cfg) -> spreading bonuses dict
    |
    v
For each candidate:
    final_activation = normalize_activation(
        compute_base_level(history, now, cfg)
    ) + spreading_bonus  # clamped to [0, 1]
    |
    v
    edge_proximity = hop_distance from spreading BFS (or 0 if not reached)
    |
    v
    score = 0.50 * semantic_similarity
          + 0.35 * final_activation
          + 0.15 * edge_proximity
    |
    v
Sort by score, return top-N
    |
    v
Record access for returned nodes (record_access)
```

### Python pseudocode for the scorer

```python
from dataclasses import dataclass


@dataclass
class ScoredNode:
    node_id: str
    score: float
    semantic_similarity: float
    activation: float
    edge_proximity: float
    # For debugging / dashboard display


def score_candidates(
    candidates: list[tuple[str, float]],  # (node_id, semantic_similarity)
    spreading_bonuses: dict[str, float],
    hop_distances: dict[str, int],
    seed_node_ids: set[str],
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
) -> list[ScoredNode]:
    """Score and rank candidate nodes."""
    results = []

    for node_id, sem_sim in candidates:
        # Compute activation (lazy -- from stored access_history)
        state = activation_states.get(node_id)
        if state and state.access_history:
            base_act = compute_activation(state.access_history, now, cfg)
        else:
            base_act = 0.0

        # Add spreading bonus, clamp to [0, 1]
        spread = spreading_bonuses.get(node_id, 0.0)
        final_act = min(1.0, base_act + spread)

        # Edge proximity
        if node_id in seed_node_ids:
            edge_prox = 1.0
        elif node_id in hop_distances:
            edge_prox = 0.5 ** hop_distances[node_id]
        else:
            edge_prox = 0.0

        # Composite score
        score = (
            cfg.weight_semantic * sem_sim
            + cfg.weight_activation * final_act
            + cfg.weight_edge_proximity * edge_prox
        )

        results.append(ScoredNode(
            node_id=node_id,
            score=score,
            semantic_similarity=sem_sim,
            activation=final_act,
            edge_proximity=edge_prox,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results
```

---

## 8. Configuration Parameters

All activation parameters are centralized in a single config object. These must be exposed in the Engram config file and validated by Pydantic.

```python
from pydantic import BaseModel, Field


class ActivationConfig(BaseModel):
    """All tunable activation engine parameters."""

    # --- ACT-R Decay ---
    decay_exponent: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="ACT-R decay exponent (d). Higher = faster forgetting. "
                    "0.5 is the standard ACT-R value.",
    )
    min_age_seconds: float = Field(
        default=1.0,
        ge=0.01,
        description="Minimum age for access timestamps to avoid division by zero.",
    )
    max_history_size: int = Field(
        default=200,
        ge=10,
        le=10000,
        description="Maximum access timestamps stored per node. "
                    "Older entries are evicted. Bounds computation cost.",
    )

    # --- Sigmoid Normalization ---
    B_mid: float = Field(
        default=-0.5,
        description="Raw B_i value that maps to activation 0.5. "
                    "Tune based on typical access patterns.",
    )
    B_scale: float = Field(
        default=1.0,
        gt=0.0,
        description="Sigmoid steepness. Larger = more gradual transition.",
    )

    # --- Spreading Activation ---
    spread_max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum BFS depth for spreading activation.",
    )
    spread_decay_per_hop: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Energy multiplier per hop. 0.5 = halve each hop.",
    )
    spread_firing_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Minimum energy to propagate to a neighbor. "
                    "Below this, the spread stops.",
    )
    spread_energy_budget: float = Field(
        default=5.0,
        gt=0.0,
        description="Total energy that can be distributed in one retrieval. "
                    "Bounds worst-case spreading cost.",
    )

    # --- Retrieval Scoring Weights ---
    weight_semantic: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight for semantic similarity in composite score.",
    )
    weight_activation: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for activation level in composite score.",
    )
    weight_edge_proximity: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for edge proximity in composite score.",
    )
    seed_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum semantic similarity for a candidate to be a "
                    "spreading activation seed.",
    )

    # --- Retrieval Limits ---
    retrieval_top_k: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of candidates from vector search before scoring.",
    )
    retrieval_top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results returned after composite scoring.",
    )
```

### Parameter rationale summary

| Parameter | Default | Why this value |
|---|---|---|
| `decay_exponent` | 0.5 | Standard ACT-R value validated across decades of cognitive modeling. |
| `max_history_size` | 200 | Balances computation cost (200 terms in the sum) with accuracy. Older accesses contribute negligibly to the power-law sum anyway. |
| `B_mid` | -0.5 | Calibrated so a node accessed once ~1 hour ago sits at activation 0.5. |
| `B_scale` | 1.0 | Gives a smooth sigmoid without extreme clipping. |
| `spread_max_hops` | 2 | Two hops captures direct associations and one level of indirect association. Three+ hops pulls in too much noise for personal-scale graphs. |
| `spread_decay_per_hop` | 0.5 | Halving per hop means hop-2 neighbors get 25% of seed energy at most. |
| `spread_firing_threshold` | 0.05 | Prevents the long tail of negligible activations from expanding the BFS frontier. |
| `spread_energy_budget` | 5.0 | With typical seed energies of 0.5-0.8, this allows spreading to ~20-30 nodes before exhaustion. |
| `weight_semantic` | 0.50 | Semantic relevance is still the primary signal. Activation is the differentiator, not the dominator. |
| `weight_activation` | 0.35 | Strong enough to re-rank results meaningfully, not so strong that cold-but-relevant results disappear. |
| `weight_edge_proximity` | 0.15 | A tiebreaker that favors structurally close nodes. Keeps associative recall working even for nodes with no access history. |

---

## 9. Lazy vs. Eager Decay

**Decision: lazy evaluation only. No background sweep.**

### How it works

Activation is never stored as a float that "decays over time." Instead:

1. We store `access_history` (list of timestamps) per node in Redis.
2. When a retrieval needs a node's activation, we call `compute_activation(access_history, now, cfg)`.
3. The power-law formula naturally produces lower values for older accesses.
4. There is no background cron job, no sweep, no stale-activation problem.

### Why not eager (background sweep)?

| Concern | Lazy | Eager (sweep) |
|---|---|---|
| Correctness | Always exact | Stale between sweeps |
| CPU cost | O(candidates * history_size) per retrieval | O(total_nodes * history_size) per sweep |
| Cold nodes | Zero cost | Still swept even if never queried |
| Complexity | Simple -- one code path | Sweep scheduler, Redis write batches, race conditions |
| Dashboard display | Compute on request | Pre-computed, possibly stale |

For a personal memory graph with ~10K nodes and 50 retrieval candidates, lazy evaluation costs 50 * 200 = 10,000 floating-point operations per retrieval -- sub-millisecond. A background sweep over 10K nodes costs 2 million operations and writes 10K Redis keys, most of which nobody reads.

### Dashboard optimization

The Activation Monitor view needs to display activation levels for many nodes at once. For this use case:

```python
def batch_compute_activations(
    node_ids: list[str],
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
) -> dict[str, float]:
    """Compute activation for a batch of nodes. Used by dashboard API."""
    return {
        nid: compute_activation(
            activation_states[nid].access_history, now, cfg
        )
        for nid in node_ids
        if nid in activation_states
    }
```

For the "top 20 most activated" leaderboard, we can maintain a Redis sorted set of `last_accessed` timestamps and only compute activation for the most recently accessed ~100 nodes, then return the top 20. This avoids scanning the full graph.

---

## 10. Contextual Boost

When a retrieval query arrives, candidates from vector search receive a contextual boost before spreading activation begins. This is the entry point for activation dynamics.

```python
def apply_contextual_boost(
    candidates: list[tuple[str, float]],  # (node_id, semantic_similarity)
    activation_states: dict[str, ActivationState],
    now: float,
    cfg: ActivationConfig,
) -> list[tuple[str, float]]:
    """
    Identify seed nodes and assign initial energy for spreading.
    Returns (node_id, energy) pairs for seeds.
    """
    seeds = []
    for node_id, sem_sim in candidates:
        if sem_sim >= cfg.seed_threshold:
            # Seed energy is proportional to semantic similarity
            energy = sem_sim * compute_activation(
                activation_states.get(node_id, ActivationState(node_id)).access_history,
                now,
                cfg,
            )
            # Floor: even cold but semantically strong nodes get some energy
            energy = max(energy, sem_sim * 0.1)
            seeds.append((node_id, energy))

    return seeds
```

Seed energy is the product of semantic similarity and current activation. This means a semantically strong AND recently active node is a powerful seed, while a semantically strong but cold node still seeds with reduced energy (the 0.1 floor). A semantically weak node is not a seed at all regardless of activation.

---

## 11. Edge Activation (Optional, Phase 2)

The original spec focuses on node activation but edges also carry activation signals. For Phase 1, edges are passive -- they have static weights used by spreading activation. In Phase 2, edges can have their own access history and activation:

- Edge activation follows the same ACT-R formula applied to edge traversal timestamps.
- Spreading activation uses `edge_weight * edge_activation` instead of just `edge_weight`.
- This lets the engine distinguish "these two nodes are connected" (static) from "this connection has been recently relevant" (dynamic).

This is deferred to avoid premature complexity. The node-only model is sufficient for launch.

---

## 12. Summary of Changes from Original Spec

| Aspect | Original | Refined |
|---|---|---|
| Decay function | Hyperbolic: `1/(1 + d*log(t+1))` | ACT-R power-law: `ln(sum(t_j^{-d}))` |
| Decay computation | Ambiguous (background sweep mentioned) | Lazy only -- compute on read |
| Stored activation | `base_activation` + `current_activation` floats | `access_history` timestamps only |
| Retrieval signals | 4 signals (semantic, activation, recency, frequency) | 3 orthogonal signals (semantic, activation, edge_proximity) |
| Scorer weights | 0.4 / 0.3 / 0.2 / 0.1 | 0.50 / 0.35 / 0.15 |
| Spreading activation | No threshold, no normalization, no visited set, no budget | Firing threshold, sqrt(degree) normalization, visited set, energy budget |
| Spreading triggers access? | Unspecified | No -- only genuine interactions record access |
| Per-node decay rate | Yes (`decay_rate` per node) | No -- single global `decay_exponent` |
| Edge activation | Not mentioned | Deferred to Phase 2 |

---

## Appendix A: Worked Example

**Setup:** Node "Python" was accessed at these times (hours ago): 2h, 8h, 24h, 72h, 168h (1 week). Node "Machine Learning" was accessed 1h ago and is connected to "Python" with edge_weight 0.7.

**Query:** "What frameworks should I use?" (semantically matches "Python" at 0.6, "Machine Learning" at 0.4)

**Step 1: Compute base activation for "Python"**

```
access ages in seconds: [7200, 28800, 86400, 259200, 604800]
d = 0.5

sum = 7200^(-0.5) + 28800^(-0.5) + 86400^(-0.5) + 259200^(-0.5) + 604800^(-0.5)
    = 0.01179 + 0.00589 + 0.00340 + 0.00196 + 0.00129
    = 0.02433

B_python = ln(0.02433) = -3.716

activation_python = sigmoid((-3.716 - (-0.5)) / 1.0)
                  = sigmoid(-3.216)
                  = 0.038
```

Python has low activation -- 5 accesses spread over a week. It's fading.

**Step 2: Compute base activation for "Machine Learning"**

```
access ages in seconds: [3600]
d = 0.5

sum = 3600^(-0.5) = 0.01667

B_ml = ln(0.01667) = -4.094

activation_ml = sigmoid((-4.094 - (-0.5)) / 1.0)
              = sigmoid(-3.594)
              = 0.027
```

ML was accessed only once 1h ago -- its activation is also modest.

**Step 3: Spreading activation**

Seeds: "Python" (sem=0.6, energy = 0.6 * max(0.038, 0.06) = 0.036), "ML" (sem=0.4, energy = 0.4 * max(0.027, 0.04) = 0.016).

Note: the floor `sem * 0.1` kicks in here because both nodes are cold.

Python spreads to ML: energy = 0.036 * 0.7 (edge_weight) * 1.0 (sqrt(1) for 1 neighbor) * 0.5 (hop decay) = 0.0126. This is below `firing_threshold` (0.05), so it does NOT propagate.

ML spreads to Python: energy = 0.016 * 0.7 * 1.0 * 0.5 = 0.0056. Also below threshold.

In this case, spreading activation has minimal effect because both seeds are cold. This is correct behavior -- the engine should not artificially inflate cold nodes just because they match a query.

**Step 4: Scoring**

```
Python:  0.50 * 0.6 + 0.35 * 0.038 + 0.15 * 1.0 = 0.300 + 0.013 + 0.15 = 0.463
ML:      0.50 * 0.4 + 0.35 * 0.027 + 0.15 * 0.5  = 0.200 + 0.009 + 0.075 = 0.284
```

Python ranks higher primarily due to semantic similarity (0.6 vs 0.4) and being a seed node (edge_proximity 1.0). If Python had been accessed 10 minutes ago instead of spread over a week, its activation would be ~0.7, and the score gap would widen to 0.545 vs 0.284.

**This is the intended behavior:** semantic similarity dominates for cold nodes; activation provides meaningful re-ranking for warm nodes.

---

## Appendix B: Decay Curve Reference

For a single access at time `t_j` with `d = 0.5`, the contribution decays as:

| Time since access | Raw contribution `t^{-0.5}` | Fraction of 1-second |
|---|---|---|
| 1 second | 1.000 | 100% |
| 1 minute | 0.129 | 12.9% |
| 1 hour | 0.0167 | 1.67% |
| 1 day | 0.0034 | 0.34% |
| 1 week | 0.0013 | 0.13% |
| 1 month | 0.0006 | 0.06% |

With multiple accesses, these contributions sum. Five accesses over a week produce a sum comparable to one access a few minutes ago -- this is how frequency compensates for recency.
