import * as THREE from "three";
import { activationColor, entityColor } from "../../lib/colors";
import type { SharedResources } from "./SharedResources";
import type { DisposalRegistry } from "./DisposalRegistry";

/**
 * Creates full neuron meshes for Focus-tier nodes (activation > 0.7).
 * Uses SharedResources for pooled geometry/materials, stores direct references
 * to soma and sprite in userData for O(1) animation access (no traverse()).
 */
export class FocusTierRenderer {
  private shared: SharedResources;
  private registry: DisposalRegistry;

  constructor(shared: SharedResources, registry: DisposalRegistry) {
    this.shared = shared;
    this.registry = registry;
  }

  /**
   * Create a full neuron Group: soma + dendrites + glow sprite.
   * Registers per-node resources in DisposalRegistry.
   */
  createNodeObject(
    nodeId: string,
    activation: number,
    entityType: string,
    accessCount: number,
    showHeatmap: boolean,
  ): THREE.Group {
    // Clean up previous resources if node is being recreated
    if (this.registry.hasNode(nodeId)) {
      this.registry.disposeNode(nodeId);
    }

    const colorHex = showHeatmap ? activationColor(activation) : entityColor(entityType);
    const color = new THREE.Color(colorHex);
    const coreRadius = 2 + activation * 6;

    // Soma — shared geometry, scaled per-node
    const somaMaterial = showHeatmap
      ? new THREE.MeshBasicMaterial({ color })
      : this.shared.getSomaMaterial(entityType);

    const somaMesh = new THREE.Mesh(this.shared.somaGeometryHigh, somaMaterial);
    somaMesh.scale.setScalar(coreRadius);

    if (showHeatmap) {
      // Heatmap materials are unique per node — track for disposal
      this.registry.trackMaterial(nodeId, somaMaterial);
    }

    const group = new THREE.Group();
    group.add(somaMesh);

    // Dendrites — unique geometry per dendrite, shared bucketed material
    const dendriteCount = Math.min(6 + accessCount, 12);
    for (let i = 0; i < dendriteCount; i++) {
      const dir = new THREE.Vector3(
        Math.random() - 0.5,
        Math.random() - 0.5,
        Math.random() - 0.5,
      ).normalize();

      const length = coreRadius * (1.5 + Math.random() * 1.5);
      const mid = dir
        .clone()
        .multiplyScalar(length * 0.5)
        .add(
          new THREE.Vector3(
            (Math.random() - 0.5) * coreRadius * 0.6,
            (Math.random() - 0.5) * coreRadius * 0.6,
            (Math.random() - 0.5) * coreRadius * 0.6,
          ),
        );
      const tip = dir.clone().multiplyScalar(length);

      const curve = new THREE.QuadraticBezierCurve3(
        new THREE.Vector3(0, 0, 0),
        mid,
        tip,
      );
      const points = curve.getPoints(6);
      const geometry = new THREE.BufferGeometry().setFromPoints(points);

      // Vertex colors fade from bright at soma to dim at tips
      const colors = new Float32Array(points.length * 3);
      for (let j = 0; j < points.length; j++) {
        const fade = 1 - j / (points.length - 1);
        colors[j * 3] = color.r * (0.3 + fade * 0.7);
        colors[j * 3 + 1] = color.g * (0.3 + fade * 0.7);
        colors[j * 3 + 2] = color.b * (0.3 + fade * 0.7);
      }
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      // Track unique geometry for disposal
      this.registry.trackGeometry(nodeId, geometry);

      const material = this.shared.getDendriteMaterial(i, dendriteCount);
      group.add(new THREE.Line(geometry, material));
    }

    // Glow sprite — shared texture, unique material (opacity varies)
    const glowAlpha = 0.04 + activation * 0.12;
    const glowSize = coreRadius * (3 + activation * 2);

    const spriteMaterial = this.shared.createGlowSpriteMaterial(color, glowAlpha);
    this.registry.trackMaterial(nodeId, spriteMaterial);

    const sprite = new THREE.Sprite(spriteMaterial);
    sprite.scale.set(glowSize, glowSize, 1);
    group.add(sprite);

    // Store direct references in userData for O(1) animation (no traverse)
    group.userData.soma = somaMesh;
    group.userData.sprite = sprite;
    group.userData.spriteMaterial = spriteMaterial;
    group.userData.baseGlowScale = glowSize;
    group.userData.baseColor = color.getHex();
    group.userData.nodeId = nodeId;
    group.userData.tier = "focus";

    this.registry.trackGroup(nodeId, group);

    return group;
  }

  /**
   * Animate breathing on a focus-tier node's sprite.
   * Called by the animation budget allocator.
   */
  animateBreathing(group: THREE.Group, time: number, idx: number): void {
    const sprite = group.userData.sprite as THREE.Sprite | undefined;
    if (!sprite) return;

    const baseScale = group.userData.baseGlowScale as number;
    const breath = 1 + 0.05 * Math.sin(time * 0.001 + idx * 0.7);
    sprite.scale.setScalar(baseScale * breath);
  }

  /**
   * Animate pulse effect on a focus-tier node.
   * Returns true if pulse is still active.
   */
  animatePulse(
    group: THREE.Group,
    fadeIntensity: number,
  ): void {
    const sprite = group.userData.sprite as THREE.Sprite | undefined;
    const soma = group.userData.soma as THREE.Mesh | undefined;

    if (sprite) {
      const baseScale = group.userData.baseGlowScale as number;
      sprite.scale.setScalar(baseScale * (1 + 1.5 * fadeIntensity));

      const mat = group.userData.spriteMaterial as THREE.SpriteMaterial;
      const baseOpacity = mat.userData?.baseOpacity ?? mat.opacity;
      mat.userData = { baseOpacity };
      mat.opacity = baseOpacity + 0.3 * fadeIntensity;
    }

    if (soma) {
      const mat = soma.material as THREE.MeshBasicMaterial;
      const baseColorHex = group.userData.baseColor as number;
      const baseColor = new THREE.Color(baseColorHex);
      const white = new THREE.Color(0xffffff);
      mat.color.copy(baseColor).lerp(white, 0.6 * fadeIntensity);
    }
  }

  /** Reset pulse visual state back to base */
  resetPulse(group: THREE.Group): void {
    const sprite = group.userData.sprite as THREE.Sprite | undefined;
    const soma = group.userData.soma as THREE.Mesh | undefined;

    if (sprite) {
      const baseScale = group.userData.baseGlowScale as number;
      sprite.scale.setScalar(baseScale);

      const mat = group.userData.spriteMaterial as THREE.SpriteMaterial;
      if (mat.userData?.baseOpacity != null) {
        mat.opacity = mat.userData.baseOpacity;
      }
    }

    if (soma) {
      const mat = soma.material as THREE.MeshBasicMaterial;
      const baseColorHex = group.userData.baseColor as number;
      if (baseColorHex != null) {
        mat.color.setHex(baseColorHex);
      }
    }
  }
}
