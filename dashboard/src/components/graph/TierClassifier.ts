/**
 * Activation-driven rendering tier classification with hysteresis.
 *
 * Three rendering tiers based on ACT-R activation scores:
 *   Focus  (>0.7)  — full neuron mesh (soma + dendrites + glow + breathing + pulse)
 *   Active (0.15-0.7) — InstancedMesh sphere (entity color, activation-scaled)
 *   Dormant (<0.15) — Points cloud (faint glow dots)
 *
 * No Deep tier — every node in the graph is always rendered.
 * Hysteresis prevents flicker at tier boundaries.
 * 300ms crossfade transitions for smooth visual handoff.
 */

export type Tier = "focus" | "active" | "dormant";

const TRANSITION_DURATION_MS = 300;

/** Hysteresis thresholds: [enterAbove, exitBelow] */
const THRESHOLDS = {
  focus:  { enter: 0.70, exit: 0.65 },
  active: { enter: 0.15, exit: 0.10 },
};

interface NodeTierState {
  currentTier: Tier;
  previousTier: Tier | null;
  transitionStart: number | null;
  /** 0 = fully previous tier, 1 = fully current tier */
  transitionProgress: number;
}

export class TierClassifier {
  private states = new Map<string, NodeTierState>();

  /** Classify a node and return its tier. Updates internal state with hysteresis. */
  classify(nodeId: string, activation: number): Tier {
    const state = this.states.get(nodeId);
    const newTier = this.computeTier(activation, state?.currentTier ?? null);

    if (!state) {
      this.states.set(nodeId, {
        currentTier: newTier,
        previousTier: null,
        transitionStart: null,
        transitionProgress: 1,
      });
      return newTier;
    }

    if (newTier !== state.currentTier) {
      state.previousTier = state.currentTier;
      state.currentTier = newTier;
      state.transitionStart = Date.now();
      state.transitionProgress = 0;
    }

    return state.currentTier;
  }

  /** Get the current tier state for a node (for transition rendering) */
  getState(nodeId: string): NodeTierState | undefined {
    return this.states.get(nodeId);
  }

  /** Advance all active transitions. Call once per frame. Returns IDs that are mid-transition. */
  updateTransitions(now: number): string[] {
    const transitioning: string[] = [];

    for (const [nodeId, state] of this.states) {
      if (state.transitionStart === null || state.transitionProgress >= 1) continue;

      const elapsed = now - state.transitionStart;
      state.transitionProgress = Math.min(1, elapsed / TRANSITION_DURATION_MS);

      if (state.transitionProgress >= 1) {
        state.previousTier = null;
        state.transitionStart = null;
      } else {
        transitioning.push(nodeId);
      }
    }

    return transitioning;
  }

  /** Get all node IDs in a specific tier */
  getNodesInTier(tier: Tier): string[] {
    const result: string[] = [];
    for (const [nodeId, state] of this.states) {
      if (state.currentTier === tier) result.push(nodeId);
    }
    return result;
  }

  /** Remove tracking for a node (when removed from graph) */
  removeNode(nodeId: string): void {
    this.states.delete(nodeId);
  }

  /** Clear all state */
  clear(): void {
    this.states.clear();
  }

  get size(): number {
    return this.states.size;
  }

  private computeTier(activation: number, currentTier: Tier | null): Tier {
    // Without existing state, use entry thresholds
    if (currentTier === null) {
      if (activation >= THRESHOLDS.focus.enter) return "focus";
      if (activation >= THRESHOLDS.active.enter) return "active";
      return "dormant";
    }

    // With hysteresis: use exit threshold for current tier, enter for others
    switch (currentTier) {
      case "focus":
        if (activation < THRESHOLDS.focus.exit) {
          if (activation >= THRESHOLDS.active.enter) return "active";
          return "dormant";
        }
        return "focus";

      case "active":
        if (activation >= THRESHOLDS.focus.enter) return "focus";
        if (activation < THRESHOLDS.active.exit) return "dormant";
        return "active";

      case "dormant":
        if (activation >= THRESHOLDS.focus.enter) return "focus";
        if (activation >= THRESHOLDS.active.enter) return "active";
        return "dormant";
    }
  }
}
