import * as THREE from "three";
import { ENTITY_TYPE_COLORS } from "../../lib/colors";

/**
 * Pooled Three.js resources shared across all nodes.
 * Instead of creating geometry/material/texture per node, we create once and reuse.
 */
export class SharedResources {
  // Geometries — 2 detail levels of icosahedron
  readonly somaGeometryHigh: THREE.IcosahedronGeometry;
  readonly somaGeometryLow: THREE.IcosahedronGeometry;
  readonly activeSphereGeometry: THREE.SphereGeometry;

  // Glow texture — single 64x64 radial gradient canvas shared by all sprites
  readonly glowTexture: THREE.CanvasTexture;

  // Per-entity-type soma materials (MeshBasicMaterial)
  readonly somaMaterials: Map<string, THREE.MeshBasicMaterial> = new Map();
  // Per-entity-type soma materials for heatmap mode (activation color)
  readonly heatmapMaterials: Map<number, THREE.MeshBasicMaterial> = new Map();

  // Dendrite materials bucketed by opacity (4 buckets)
  readonly dendriteMaterials: THREE.LineBasicMaterial[] = [];

  private disposed = false;

  constructor() {
    // Soma geometries — all nodes share these, scale via mesh.scale
    this.somaGeometryHigh = new THREE.IcosahedronGeometry(1, 1);
    this.somaGeometryLow = new THREE.IcosahedronGeometry(1, 0);
    this.activeSphereGeometry = new THREE.SphereGeometry(1, 8, 6);

    // Shared glow texture
    this.glowTexture = this.createGlowTexture();

    // Build soma materials for each known entity type
    for (const [type, hex] of Object.entries(ENTITY_TYPE_COLORS)) {
      if (this.somaMaterials.has(type)) continue;
      this.somaMaterials.set(
        type,
        new THREE.MeshBasicMaterial({ color: new THREE.Color(hex) }),
      );
    }

    // 4 dendrite opacity buckets: 0.40, 0.33, 0.28, 0.25
    const opacities = [0.40, 0.33, 0.28, 0.25];
    for (const opacity of opacities) {
      this.dendriteMaterials.push(
        new THREE.LineBasicMaterial({
          vertexColors: true,
          transparent: true,
          opacity,
          depthWrite: false,
        }),
      );
    }
  }

  /** Get soma material for an entity type (returns shared instance) */
  getSomaMaterial(entityType: string): THREE.MeshBasicMaterial {
    const existing = this.somaMaterials.get(entityType);
    if (existing) return existing;
    // Fallback: create for unknown type
    const mat = new THREE.MeshBasicMaterial({
      color: new THREE.Color(ENTITY_TYPE_COLORS["Other"] ?? "#94a3b8"),
    });
    this.somaMaterials.set(entityType, mat);
    return mat;
  }

  /** Get dendrite material by bucket index (0-3 based on dendrite position) */
  getDendriteMaterial(dendriteIndex: number, totalDendrites: number): THREE.LineBasicMaterial {
    const ratio = totalDendrites > 1 ? dendriteIndex / (totalDendrites - 1) : 0;
    const bucket = Math.min(3, Math.floor(ratio * 4));
    return this.dendriteMaterials[bucket];
  }

  /** Create a SpriteMaterial that uses the shared glow texture (per-node for opacity control) */
  createGlowSpriteMaterial(color: THREE.Color, opacity: number): THREE.SpriteMaterial {
    return new THREE.SpriteMaterial({
      map: this.glowTexture,
      color,
      transparent: true,
      opacity,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
  }

  private createGlowTexture(): THREE.CanvasTexture {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
      gradient.addColorStop(0, "rgba(255, 255, 255, 1)");
      gradient.addColorStop(0.2, "rgba(255, 255, 255, 0.5)");
      gradient.addColorStop(0.5, "rgba(255, 255, 255, 0.12)");
      gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 64, 64);
    }
    return new THREE.CanvasTexture(canvas);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;

    this.somaGeometryHigh.dispose();
    this.somaGeometryLow.dispose();
    this.activeSphereGeometry.dispose();
    this.glowTexture.dispose();

    for (const mat of this.somaMaterials.values()) mat.dispose();
    for (const mat of this.heatmapMaterials.values()) mat.dispose();
    for (const mat of this.dendriteMaterials) mat.dispose();

    this.somaMaterials.clear();
    this.heatmapMaterials.clear();
  }
}
