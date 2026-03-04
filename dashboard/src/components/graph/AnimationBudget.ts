/**
 * Per-frame animation work budget.
 * Caps the number of nodes animated each frame to maintain 60fps.
 */

const MAX_ANIMATED_PER_FRAME = 50;

export class AnimationBudget {
  /** Round-robin cursor for idle Focus-tier nodes */
  private roundRobinIndex = 0;

  /**
   * Allocate animation budget for this frame.
   * Pulsing nodes always get priority, remaining budget filled via round-robin
   * across idle Focus-tier nodes.
   *
   * @returns Array of nodeIds to animate this frame
   */
  allocate(pulsingNodeIds: string[], focusTierNodeIds: string[]): string[] {
    const result: string[] = [];

    // Priority: all pulsing nodes (up to budget)
    const pulseCount = Math.min(pulsingNodeIds.length, MAX_ANIMATED_PER_FRAME);
    for (let i = 0; i < pulseCount; i++) {
      result.push(pulsingNodeIds[i]);
    }

    // Fill remaining budget with idle focus-tier nodes via round-robin
    const remaining = MAX_ANIMATED_PER_FRAME - result.length;
    if (remaining <= 0 || focusTierNodeIds.length === 0) return result;

    const pulsingSet = new Set(pulsingNodeIds);
    const idleNodes = focusTierNodeIds.filter((id) => !pulsingSet.has(id));

    if (idleNodes.length === 0) return result;

    // Wrap round-robin index
    if (this.roundRobinIndex >= idleNodes.length) {
      this.roundRobinIndex = 0;
    }

    const count = Math.min(remaining, idleNodes.length);
    for (let i = 0; i < count; i++) {
      const idx = (this.roundRobinIndex + i) % idleNodes.length;
      result.push(idleNodes[idx]);
    }

    this.roundRobinIndex = (this.roundRobinIndex + count) % idleNodes.length;

    return result;
  }

  /** Reset round-robin state */
  reset(): void {
    this.roundRobinIndex = 0;
  }
}
