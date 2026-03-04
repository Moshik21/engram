import * as THREE from "three";
import { entityColor } from "../../lib/colors";
import type { SharedResources } from "./SharedResources";

const MAX_INSTANCES = 512;
const DUMMY_MATRIX = new THREE.Matrix4();
const TEMP_COLOR = new THREE.Color();

/**
 * Batch renderer for Active-tier nodes using THREE.InstancedMesh.
 * 1 draw call for up to 512 spheres. Added directly to the scene,
 * reads positions from force-graph's node data array.
 */
export class ActiveTierRenderer {
  readonly mesh: THREE.InstancedMesh;
  /** Maps nodeId → instance slot index */
  private slotMap = new Map<string, number>();
  /** Free slot indices for reuse */
  private freeSlots: number[] = [];
  private activeCount = 0;

  constructor(shared: SharedResources) {
    const material = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.85,
    });

    this.mesh = new THREE.InstancedMesh(
      shared.activeSphereGeometry,
      material,
      MAX_INSTANCES,
    );
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Enable per-instance color
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(MAX_INSTANCES * 3),
      3,
    );
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    this.mesh.count = 0;
    this.mesh.frustumCulled = false;
  }

  /** Add a node to the instanced batch */
  addNode(nodeId: string, entityType: string): number | null {
    if (this.slotMap.has(nodeId)) return this.slotMap.get(nodeId)!;
    if (this.activeCount >= MAX_INSTANCES && this.freeSlots.length === 0) return null;

    const slot = this.freeSlots.length > 0
      ? this.freeSlots.pop()!
      : this.activeCount++;

    this.slotMap.set(nodeId, slot);

    // Set instance color from entity type
    TEMP_COLOR.set(entityColor(entityType));
    this.mesh.instanceColor!.setXYZ(slot, TEMP_COLOR.r, TEMP_COLOR.g, TEMP_COLOR.b);
    this.mesh.instanceColor!.needsUpdate = true;

    this.updateDrawCount();
    return slot;
  }

  /** Remove a node from the batch, freeing its slot */
  removeNode(nodeId: string): void {
    const slot = this.slotMap.get(nodeId);
    if (slot === undefined) return;

    // Hide by zeroing the scale
    DUMMY_MATRIX.makeScale(0, 0, 0);
    this.mesh.setMatrixAt(slot, DUMMY_MATRIX);

    this.slotMap.delete(nodeId);
    this.freeSlots.push(slot);
    this.mesh.instanceMatrix.needsUpdate = true;
    this.updateDrawCount();
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
    activationMap: Map<string, number>,
  ): void {
    let needsUpdate = false;

    for (const [nodeId, slot] of this.slotMap) {
      const data = nodeDataMap.get(nodeId);
      if (!data) continue;

      const x = data.x ?? 0;
      const y = data.y ?? 0;
      const z = data.z ?? 0;
      const activation = activationMap.get(nodeId) ?? 0.3;
      const scale = 3 + activation * 5;

      DUMMY_MATRIX.makeScale(scale, scale, scale);
      DUMMY_MATRIX.setPosition(x, y, z);
      this.mesh.setMatrixAt(slot, DUMMY_MATRIX);
      needsUpdate = true;
    }

    if (needsUpdate) {
      this.mesh.instanceMatrix.needsUpdate = true;
    }
  }

  /** Number of active instances */
  get count(): number {
    return this.slotMap.size;
  }

  /** Dispose the instanced mesh */
  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.slotMap.clear();
    this.freeSlots = [];
    this.activeCount = 0;
  }

  private updateDrawCount(): void {
    // count must be high enough to include all occupied slots
    let maxSlot = -1;
    for (const slot of this.slotMap.values()) {
      if (slot > maxSlot) maxSlot = slot;
    }
    this.mesh.count = maxSlot + 1;
  }
}
