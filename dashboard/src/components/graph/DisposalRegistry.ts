import * as THREE from "three";

interface NodeResources {
  /** Unique geometries owned by this node (dendrite curves) */
  geometries: THREE.BufferGeometry[];
  /** Unique materials owned by this node (sprite material) */
  materials: THREE.Material[];
  /** The root Group returned by nodeThreeObject */
  group: THREE.Group | null;
}

/**
 * Tracks per-node Three.js resources for proper disposal.
 * Shared resources (from SharedResources) are NOT tracked here — they have their own lifecycle.
 */
export class DisposalRegistry {
  private nodes = new Map<string, NodeResources>();

  /** Register a unique geometry for a node */
  trackGeometry(nodeId: string, geometry: THREE.BufferGeometry): void {
    this.getOrCreate(nodeId).geometries.push(geometry);
  }

  /** Register a unique material for a node */
  trackMaterial(nodeId: string, material: THREE.Material): void {
    this.getOrCreate(nodeId).materials.push(material);
  }

  /** Register the root Group for a node */
  trackGroup(nodeId: string, group: THREE.Group): void {
    this.getOrCreate(nodeId).group = group;
  }

  /** Dispose all unique resources for a single node */
  disposeNode(nodeId: string): void {
    const resources = this.nodes.get(nodeId);
    if (!resources) return;

    for (const geo of resources.geometries) geo.dispose();
    for (const mat of resources.materials) mat.dispose();

    // Remove children from group but don't dispose group itself
    // (react-force-graph manages the Group's scene attachment)
    if (resources.group) {
      while (resources.group.children.length > 0) {
        resources.group.remove(resources.group.children[0]);
      }
    }

    this.nodes.delete(nodeId);
  }

  /** Check if a node has tracked resources */
  hasNode(nodeId: string): boolean {
    return this.nodes.has(nodeId);
  }

  /** Dispose all tracked resources (unmount cleanup) */
  disposeAll(): void {
    for (const nodeId of this.nodes.keys()) {
      this.disposeNode(nodeId);
    }
  }

  /** Number of tracked nodes */
  get size(): number {
    return this.nodes.size;
  }

  private getOrCreate(nodeId: string): NodeResources {
    let entry = this.nodes.get(nodeId);
    if (!entry) {
      entry = { geometries: [], materials: [], group: null };
      this.nodes.set(nodeId, entry);
    }
    return entry;
  }
}
