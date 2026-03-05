/**
 * Level-of-Detail controller for semantic zoom.
 *
 * Tracks camera distance from the graph centroid and determines the current
 * zoom tier. Each tier controls:
 *   - How many nodes are visible (visibility budget)
 *   - Node rendering complexity
 *   - Edge visibility
 *   - Label visibility
 *
 * Brain analogy:
 *   Macro      → Hemispheres (colored blobs per cluster)
 *   Region     → Brain lobes (hub entities visible, rest as dots)
 *   Neighborhood → Gyri/folds (most entities, activation threshold)
 *   Detail     → Neurons (full labels, all edges, sparklines)
 *   Synapse    → Single entity expanded (temporal history, edge weights)
 *
 * Transitions use sigmoid interpolation to avoid visual snapping.
 */

export type ZoomTier = "macro" | "region" | "neighborhood" | "detail" | "synapse";

export interface LODConfig {
  /** Max nodes to render at this tier (rest hidden) */
  visibilityBudget: number;
  /** Min activation for a node to be visible (lower = show more) */
  activationFloor: number;
  /** Whether to show edge labels */
  showLabels: boolean;
  /** Whether to show edges */
  showEdges: boolean;
  /** Node detail level: 0 = point, 1 = sphere, 2 = full neuron */
  detailLevel: number;
}

/** Camera distance thresholds for each tier (with hysteresis) */
interface TierThreshold {
  enter: number;  // Distance to enter this tier (zooming in)
  exit: number;   // Distance to exit this tier (zooming out)
}

const TIER_THRESHOLDS: Record<ZoomTier, TierThreshold> = {
  synapse:      { enter: 80,   exit: 100 },
  detail:       { enter: 200,  exit: 250 },
  neighborhood: { enter: 500,  exit: 600 },
  region:       { enter: 1200, exit: 1400 },
  macro:        { enter: Infinity, exit: Infinity },
};

const TIER_CONFIGS: Record<ZoomTier, LODConfig> = {
  macro: {
    visibilityBudget: 100,
    activationFloor: 0.0,
    showLabels: false,
    showEdges: false,
    detailLevel: 0,
  },
  region: {
    visibilityBudget: 500,
    activationFloor: 0.0,
    showLabels: false,
    showEdges: true,
    detailLevel: 0,
  },
  neighborhood: {
    visibilityBudget: 2000,
    activationFloor: 0.0,
    showLabels: false,
    showEdges: true,
    detailLevel: 1,
  },
  detail: {
    visibilityBudget: 5000,
    activationFloor: 0.0,
    showLabels: true,
    showEdges: true,
    detailLevel: 2,
  },
  synapse: {
    visibilityBudget: Infinity,
    activationFloor: 0.0,
    showLabels: true,
    showEdges: true,
    detailLevel: 2,
  },
};

/** Ordered from closest to farthest */
const TIER_ORDER: ZoomTier[] = ["synapse", "detail", "neighborhood", "region", "macro"];

export class LODController {
  private currentTier: ZoomTier = "neighborhood";
  private cameraDistance = 500;
  /** Smoothed transition value (0-1) for gradual opacity/scale changes */
  private transitionAlpha = 1;
  private lastTierChangeTime = 0;

  /** Graph centroid (updated periodically from node positions) */
  private centroidX = 0;
  private centroidY = 0;
  private centroidZ = 0;

  /** Current visibility set — sorted by activation, capped by budget */
  private visibleNodeIds = new Set<string>();
  private visibilityDirty = true;

  get tier(): ZoomTier {
    return this.currentTier;
  }

  get config(): LODConfig {
    return TIER_CONFIGS[this.currentTier];
  }

  get distance(): number {
    return this.cameraDistance;
  }

  get alpha(): number {
    return this.transitionAlpha;
  }

  /**
   * Update camera distance and recalculate zoom tier.
   * Call once per frame from the animation loop.
   */
  updateCamera(cameraX: number, cameraY: number, cameraZ: number, now: number): boolean {
    const dx = cameraX - this.centroidX;
    const dy = cameraY - this.centroidY;
    const dz = cameraZ - this.centroidZ;
    this.cameraDistance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const newTier = this.computeTier(this.cameraDistance);
    const changed = newTier !== this.currentTier;

    if (changed) {
      this.currentTier = newTier;
      this.lastTierChangeTime = now;
      this.transitionAlpha = 0;
      this.visibilityDirty = true;
    }

    // Smooth transition over 300ms
    if (this.transitionAlpha < 1) {
      const elapsed = now - this.lastTierChangeTime;
      this.transitionAlpha = Math.min(1, elapsed / 300);
    }

    return changed;
  }

  /**
   * Update graph centroid from node positions.
   * Call periodically (not every frame — every ~500ms is fine).
   */
  updateCentroid(
    nodes: Array<{ x?: number; y?: number; z?: number }>,
  ): void {
    if (nodes.length === 0) return;

    let sx = 0, sy = 0, sz = 0;
    let count = 0;

    // Sample up to 200 nodes for centroid (avoid iterating all 50k)
    const step = Math.max(1, Math.floor(nodes.length / 200));
    for (let i = 0; i < nodes.length; i += step) {
      const n = nodes[i];
      if (n.x != null) {
        sx += n.x;
        sy += n.y ?? 0;
        sz += n.z ?? 0;
        count++;
      }
    }

    if (count > 0) {
      this.centroidX = sx / count;
      this.centroidY = sy / count;
      this.centroidZ = sz / count;
    }
  }

  /**
   * Rebuild the visibility set.
   * Ranks nodes by activation, caps by budget, respects activation floor.
   * Returns true if visibility changed.
   */
  rebuildVisibility(
    activations: Map<string, number>,
    allNodeIds: string[],
  ): boolean {
    if (!this.visibilityDirty) return false;
    this.visibilityDirty = false;

    const cfg = TIER_CONFIGS[this.currentTier];
    const floor = cfg.activationFloor;
    const budget = cfg.visibilityBudget;

    // Fast path: budget exceeds node count and no floor filter
    if (budget >= allNodeIds.length && floor <= 0) {
      if (this.visibleNodeIds.size === allNodeIds.length) return false;
      this.visibleNodeIds = new Set(allNodeIds);
      return true;
    }

    // Filter by activation floor
    const candidates: Array<{ id: string; activation: number }> = [];
    for (const id of allNodeIds) {
      const a = activations.get(id) ?? 0;
      if (a >= floor) {
        candidates.push({ id, activation: a });
      }
    }

    // If under budget, all candidates pass
    if (candidates.length <= budget) {
      const newSet = new Set(candidates.map((c) => c.id));
      const changed = newSet.size !== this.visibleNodeIds.size ||
        [...newSet].some((id) => !this.visibleNodeIds.has(id));
      this.visibleNodeIds = newSet;
      return changed;
    }

    // Over budget: take top-N by activation
    candidates.sort((a, b) => b.activation - a.activation);
    const newSet = new Set<string>();
    for (let i = 0; i < budget && i < candidates.length; i++) {
      newSet.add(candidates[i].id);
    }

    const changed = newSet.size !== this.visibleNodeIds.size ||
      [...newSet].some((id) => !this.visibleNodeIds.has(id));
    this.visibleNodeIds = newSet;
    return changed;
  }

  /** Check if a node is in the current visibility set */
  isVisible(nodeId: string): boolean {
    // Before first rebuildVisibility call, show all nodes
    if (this.visibleNodeIds.size === 0 && this.visibilityDirty) return true;
    return this.visibleNodeIds.has(nodeId);
  }

  /** Force visibility recalculation next frame */
  invalidateVisibility(): void {
    this.visibilityDirty = true;
  }

  /** Get detail level for a specific node based on distance + activation */
  getNodeDetailLevel(_nodeId: string, activation: number): number {
    const base = TIER_CONFIGS[this.currentTier].detailLevel;
    // High-activation nodes get promoted one detail level at medium zoom
    if (base < 2 && activation >= 0.7) return Math.min(2, base + 1);
    return base;
  }

  private computeTier(distance: number): ZoomTier {
    // Check from closest to farthest with hysteresis
    for (const tier of TIER_ORDER) {
      if (tier === "macro") return "macro"; // fallback

      const threshold = TIER_THRESHOLDS[tier];
      const isCurrentTier = tier === this.currentTier;

      // Use exit threshold if we're currently in this tier (resist leaving)
      // Use enter threshold if we're trying to enter
      const limit = isCurrentTier ? threshold.exit : threshold.enter;

      if (distance <= limit) return tier;
    }

    return "macro";
  }
}
