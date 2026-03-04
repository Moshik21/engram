import * as THREE from "three";
import { entityColor } from "../../lib/colors";

const MAX_POINTS = 8192;

/**
 * Batch renderer for Dormant-tier nodes using THREE.Points.
 * 1 draw call for thousands of faint glow dots.
 * Uses setDrawRange to render only occupied slots.
 */
export class DormantTierRenderer {
  readonly points: THREE.Points;
  private positions: Float32Array;
  private colors: Float32Array;
  /** Maps nodeId → slot index */
  private slotMap = new Map<string, number>();
  /** Free slot indices for reuse */
  private freeSlots: number[] = [];
  private nextSlot = 0;

  constructor() {
    this.positions = new Float32Array(MAX_POINTS * 3);
    this.colors = new Float32Array(MAX_POINTS * 3);

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    geometry.setDrawRange(0, 0);

    const material = new THREE.PointsMaterial({
      size: 6,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      vertexColors: true,
    });

    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
  }

  /** Add a node to the points cloud */
  addNode(nodeId: string, entityType: string): number | null {
    if (this.slotMap.has(nodeId)) return this.slotMap.get(nodeId)!;
    if (this.nextSlot >= MAX_POINTS && this.freeSlots.length === 0) return null;

    const slot = this.freeSlots.length > 0 ? this.freeSlots.pop()! : this.nextSlot++;
    this.slotMap.set(nodeId, slot);

    // Set vertex color from entity type
    const color = new THREE.Color(entityColor(entityType));
    this.colors[slot * 3] = color.r;
    this.colors[slot * 3 + 1] = color.g;
    this.colors[slot * 3 + 2] = color.b;
    this.points.geometry.attributes.color.needsUpdate = true;

    this.updateDrawRange();
    return slot;
  }

  /** Remove a node, freeing its slot */
  removeNode(nodeId: string): void {
    const slot = this.slotMap.get(nodeId);
    if (slot === undefined) return;

    // Move to origin (invisible when at 0,0,0 with tiny point)
    this.positions[slot * 3] = 0;
    this.positions[slot * 3 + 1] = 0;
    this.positions[slot * 3 + 2] = 0;
    this.colors[slot * 3] = 0;
    this.colors[slot * 3 + 1] = 0;
    this.colors[slot * 3 + 2] = 0;

    this.slotMap.delete(nodeId);
    this.freeSlots.push(slot);
    this.points.geometry.attributes.position.needsUpdate = true;
    this.points.geometry.attributes.color.needsUpdate = true;
    this.updateDrawRange();
  }

  /** Check if a node is in this renderer */
  hasNode(nodeId: string): boolean {
    return this.slotMap.has(nodeId);
  }

  /**
   * Sync positions from force-graph node data.
   * Called each frame from the animation loop.
   */
  updatePositions(
    nodeDataMap: Map<string, { x?: number; y?: number; z?: number }>,
  ): void {
    let needsUpdate = false;

    for (const [nodeId, slot] of this.slotMap) {
      const data = nodeDataMap.get(nodeId);
      if (!data) continue;

      this.positions[slot * 3] = data.x ?? 0;
      this.positions[slot * 3 + 1] = data.y ?? 0;
      this.positions[slot * 3 + 2] = data.z ?? 0;
      needsUpdate = true;
    }

    if (needsUpdate) {
      this.points.geometry.attributes.position.needsUpdate = true;
    }
  }

  /** Number of active points */
  get count(): number {
    return this.slotMap.size;
  }

  /** Dispose the points cloud */
  dispose(): void {
    this.points.geometry.dispose();
    (this.points.material as THREE.Material).dispose();
    this.slotMap.clear();
    this.freeSlots = [];
    this.nextSlot = 0;
  }

  private updateDrawRange(): void {
    // Draw range must cover all occupied slots
    let maxSlot = -1;
    for (const slot of this.slotMap.values()) {
      if (slot > maxSlot) maxSlot = slot;
    }
    this.points.geometry.setDrawRange(0, maxSlot + 1);
  }
}
