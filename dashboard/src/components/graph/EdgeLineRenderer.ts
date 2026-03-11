import * as THREE from "three";

const MAX_EDGES = 16384;

/**
 * Single-draw-call edge renderer using THREE.LineSegments.
 * Pre-allocates buffers for up to 16,384 edges.
 * Each edge = 2 vertices = 6 floats (position) + 6 floats (color).
 */
export class EdgeLineRenderer {
  readonly lineSegments: THREE.LineSegments;
  private positions: Float32Array;
  private colors: Float32Array;
  /** Maps "source\0target" → slot index (each slot = 2 vertices) */
  private slotMap = new Map<string, number>();
  private edgeCount = 0;

  constructor() {
    this.positions = new Float32Array(MAX_EDGES * 2 * 3);
    this.colors = new Float32Array(MAX_EDGES * 2 * 3);

    // Default color: dim teal (dark enough to not blow out in dense clusters)
    const defaultR = 40 / 255;
    const defaultG = 90 / 255;
    const defaultB = 140 / 255;
    for (let i = 0; i < MAX_EDGES * 2; i++) {
      this.colors[i * 3] = defaultR;
      this.colors[i * 3 + 1] = defaultG;
      this.colors[i * 3 + 2] = defaultB;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.positions, 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(this.colors, 3),
    );
    geometry.setDrawRange(0, 0);

    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
    });

    this.lineSegments = new THREE.LineSegments(geometry, material);
    this.lineSegments.frustumCulled = false;
  }

  /**
   * Rebuild the edge index from the current link list.
   * Call when graphData changes.
   */
  rebuild(
    links: Array<{
      source?: string | { id?: string };
      target?: string | { id?: string };
    }>,
  ): void {
    this.slotMap.clear();
    this.edgeCount = 0;

    const count = Math.min(links.length, MAX_EDGES);
    for (let i = 0; i < count; i++) {
      const link = links[i];
      const sourceId = typeof link.source === "string"
        ? link.source
        : link.source?.id ?? "";
      const targetId = typeof link.target === "string"
        ? link.target
        : link.target?.id ?? "";
      if (!sourceId || !targetId) continue;

      const key = `${sourceId}\0${targetId}`;
      this.slotMap.set(key, this.edgeCount);
      this.edgeCount++;
    }

    // Reset default colors for active edges
    const defaultR = 40 / 255;
    const defaultG = 90 / 255;
    const defaultB = 140 / 255;
    for (let i = 0; i < this.edgeCount * 2; i++) {
      this.colors[i * 3] = defaultR;
      this.colors[i * 3 + 1] = defaultG;
      this.colors[i * 3 + 2] = defaultB;
    }
    this.lineSegments.geometry.attributes.color.needsUpdate = true;

    this.lineSegments.geometry.setDrawRange(0, this.edgeCount * 2);
  }

  /**
   * Update endpoint positions each frame from node position data.
   */
  updatePositions(
    nodeDataMap: Map<string, Record<string, unknown>>,
  ): void {
    let needsUpdate = false;

    for (const [key, slot] of this.slotMap) {
      const sepIdx = key.indexOf("\0");
      const sourceId = key.slice(0, sepIdx);
      const targetId = key.slice(sepIdx + 1);

      const srcNode = nodeDataMap.get(sourceId);
      const tgtNode = nodeDataMap.get(targetId);
      if (!srcNode || !tgtNode) continue;

      const base = slot * 6; // 2 vertices * 3 components
      this.positions[base] = (srcNode.x as number) ?? 0;
      this.positions[base + 1] = (srcNode.y as number) ?? 0;
      this.positions[base + 2] = (srcNode.z as number) ?? 0;
      this.positions[base + 3] = (tgtNode.x as number) ?? 0;
      this.positions[base + 4] = (tgtNode.y as number) ?? 0;
      this.positions[base + 5] = (tgtNode.z as number) ?? 0;
      needsUpdate = true;
    }

    if (needsUpdate) {
      this.lineSegments.geometry.attributes.position.needsUpdate = true;
    }
  }

  /**
   * Set per-edge color for pulse highlighting.
   * key format: "sourceId\0targetId"
   */
  setEdgeColor(key: string, r: number, g: number, b: number): void {
    const slot = this.slotMap.get(key);
    if (slot === undefined) return;

    const base = slot * 6;
    // Both vertices get the same color
    this.colors[base] = r;
    this.colors[base + 1] = g;
    this.colors[base + 2] = b;
    this.colors[base + 3] = r;
    this.colors[base + 4] = g;
    this.colors[base + 5] = b;

    this.lineSegments.geometry.attributes.color.needsUpdate = true;
  }

  /** Set overall edge opacity */
  setOpacity(opacity: number): void {
    const mat = this.lineSegments.material as THREE.LineBasicMaterial;
    mat.opacity = opacity;
    mat.transparent = true;
  }

  /** Number of active edges */
  get count(): number {
    return this.edgeCount;
  }

  /** Dispose all GPU resources */
  dispose(): void {
    this.lineSegments.geometry.dispose();
    (this.lineSegments.material as THREE.Material).dispose();
    this.slotMap.clear();
    this.edgeCount = 0;
  }
}
