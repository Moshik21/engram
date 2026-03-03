import { useCallback, useMemo, useRef, useEffect } from "react";
import ForceGraph3D from "react-force-graph-3d";
import ForceGraph2D from "react-force-graph-2d";
import * as THREE from "three";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { useEngramStore } from "../store";
import { activationColor, entityColor } from "../lib/colors";
import { NodeTooltip } from "./NodeTooltip";
import { EmptyState } from "./EmptyState";

/**
 * Create a biologically-inspired neuron node: soma + dendrite tendrils + glow.
 */
function createNodeObject(node: Record<string, unknown>, showHeatmap: boolean) {
  const activation = (node.activationCurrent as number) ?? 0;
  const type = (node.entityType as string) ?? "Other";
  const accessCount = (node.accessCount as number) ?? 0;

  const colorHex = showHeatmap ? activationColor(activation) : entityColor(type);
  const color = new THREE.Color(colorHex);

  // Core soma: faceted icosahedron for organic feel
  const coreRadius = 2 + activation * 6;
  const somaGeometry = new THREE.IcosahedronGeometry(coreRadius, 1);
  const somaMaterial = new THREE.MeshStandardMaterial({
    color: color.clone().multiplyScalar(0.4),
    emissive: color,
    emissiveIntensity: 0.3 + activation * 0.7,
    roughness: 0.6,
    metalness: 0.1,
    transparent: true,
    opacity: 0.9,
  });
  const somaMesh = new THREE.Mesh(somaGeometry, somaMaterial);

  const group = new THREE.Group();
  group.add(somaMesh);

  // Dendrite tendrils — thin lines radiating outward
  const dendriteCount = Math.min(6 + accessCount, 12);
  for (let i = 0; i < dendriteCount; i++) {
    const dir = new THREE.Vector3(
      Math.random() - 0.5,
      Math.random() - 0.5,
      Math.random() - 0.5,
    ).normalize();

    const length = coreRadius * (1.5 + Math.random() * 1.5);

    // 3-point curved path: start → random midpoint → tip
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

    const tipOpacity = 0.05 + Math.random() * 0.1;
    const material = new THREE.LineBasicMaterial({
      color: color,
      transparent: true,
      opacity: 0.3 + (1 - i / dendriteCount) * 0.3,
      depthWrite: false,
    });

    // Assign varying opacity per vertex via vertex colors
    const colors = new Float32Array(points.length * 3);
    for (let j = 0; j < points.length; j++) {
      const fade = 1 - j / (points.length - 1);
      const r = color.r * (0.3 + fade * 0.7);
      const g = color.g * (0.3 + fade * 0.7);
      const b = color.b * (0.3 + fade * 0.7);
      colors[j * 3] = r;
      colors[j * 3 + 1] = g;
      colors[j * 3 + 2] = b;
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    material.vertexColors = true;
    // Override opacity for tip fade
    material.opacity = 0.15 + tipOpacity;

    const line = new THREE.Line(geometry, material);
    group.add(line);
  }

  // Outer glow sprite — 4-stop gradient for softer bioluminescent halo
  const glowAlpha = 0.1 + activation * 0.3;
  const glowSize = coreRadius * (3.5 + activation * 2.5);

  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
  gradient.addColorStop(0, `rgba(255, 255, 255, ${glowAlpha})`);
  gradient.addColorStop(0.2, `rgba(255, 255, 255, ${glowAlpha * 0.5})`);
  gradient.addColorStop(0.5, `rgba(255, 255, 255, ${glowAlpha * 0.15})`);
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 64, 64);

  const texture = new THREE.CanvasTexture(canvas);
  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    color: color,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const sprite = new THREE.Sprite(spriteMaterial);
  sprite.scale.set(glowSize, glowSize, 1);

  group.add(sprite);

  return group;
}

const PULSE_DURATION_MS = 1500;
const CASCADE_DELAY_MS = 300;
const CASCADE_INTENSITY = 0.6;
const CASCADE_MAX_QUEUE = 100;

interface CascadeEntry {
  targetId: string;
  fireAt: number;
  intensity: number;
}

export function GraphExplorer() {
  const nodes = useEngramStore((s) => s.nodes);
  const edges = useEngramStore((s) => s.edges);
  const isLoading = useEngramStore((s) => s.isLoading);
  const renderMode = useEngramStore((s) => s.renderMode);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const showEdgeLabels = useEngramStore((s) => s.showEdgeLabels);
  const selectNode = useEngramStore((s) => s.selectNode);
  const hoverNode = useEngramStore((s) => s.hoverNode);
  const expandNode = useEngramStore((s) => s.expandNode);
  const fgRef = useRef<{
    d3Force: (name: string) => unknown;
    emitParticle?: (link: unknown) => void;
    scene?: () => THREE.Scene;
    renderer?: () => THREE.WebGLRenderer;
    camera?: () => THREE.Camera;
    postProcessingComposer?: () => unknown;
  } | null>(null);

  // ForceGraph mutates node/edge objects (adds __threeObj, x, y, z, etc.)
  // so we must give it mutable shallow copies, not frozen Immer proxies.
  // Also assign random curve rotation per link for organic 3D curves.
  const graphData = useMemo(() => {
    const nodeList = Object.values(nodes).map((n) => ({ ...n }));
    const edgeList = Object.values(edges).map((e) => ({
      ...e,
      __curveRotation: Math.random() * Math.PI * 2,
    }));
    return { nodes: nodeList, links: edgeList };
  }, [nodes, edges]);

  // Tune force simulation for better layout
  useEffect(() => {
    if (fgRef.current && renderMode === "3d") {
      const fg = fgRef.current as {
        d3Force: (
          name: string,
        ) =>
          | {
              strength?: (v: number) => unknown;
              distanceMax?: (v: number) => unknown;
            }
          | undefined;
      };
      const charge = fg.d3Force("charge");
      if (charge?.strength) charge.strength(-40);
      if (charge?.distanceMax) charge.distanceMax(300);
    }
  }, [graphData, renderMode]);

  // ── Fog & ambient lighting setup ──
  useEffect(() => {
    if (renderMode !== "3d") return;

    const checkScene = () => {
      const fg = fgRef.current as {
        scene?: () => THREE.Scene;
      } | null;
      if (!fg?.scene) return false;

      const scene = fg.scene();

      // Exponential fog — deep indigo fading
      if (!scene.fog) {
        scene.fog = new THREE.FogExp2(0x0a0a1a, 0.003);
      }

      // Add ambient light for MeshStandardMaterial
      const hasAmbient = scene.children.some(
        (c) => c instanceof THREE.AmbientLight,
      );
      if (!hasAmbient) {
        scene.add(new THREE.AmbientLight(0x334466, 0.6));
        const pointLight = new THREE.PointLight(0x6688cc, 0.8, 500);
        pointLight.position.set(0, 0, 0);
        scene.add(pointLight);
      }

      return true;
    };

    // ForceGraph3D may not have scene ready immediately
    if (!checkScene()) {
      const id = setInterval(() => {
        if (checkScene()) clearInterval(id);
      }, 100);
      return () => clearInterval(id);
    }
  }, [renderMode]);

  // ── UnrealBloomPass — bioluminescent glow ──
  const bloomAddedRef = useRef(false);

  useEffect(() => {
    if (renderMode !== "3d") {
      bloomAddedRef.current = false;
      return;
    }

    const addBloom = () => {
      const fg = fgRef.current as {
        renderer?: () => THREE.WebGLRenderer;
        scene?: () => THREE.Scene;
        camera?: () => THREE.Camera;
        postProcessingComposer?: () => {
          passes: unknown[];
          addPass: (pass: unknown) => void;
        };
      } | null;
      if (!fg?.renderer || !fg?.scene || !fg?.camera) return false;

      if (bloomAddedRef.current) return true;

      try {
        const renderer = fg.renderer();
        const bloomPass = new UnrealBloomPass(
          new THREE.Vector2(
            renderer.domElement.width,
            renderer.domElement.height,
          ),
          1.2, // strength
          0.6, // radius
          0.3, // threshold
        );

        // Use the built-in post-processing composer if available
        if (fg.postProcessingComposer) {
          const composer = fg.postProcessingComposer();
          composer.addPass(bloomPass);
        }

        bloomAddedRef.current = true;
      } catch {
        // Post-processing may not be ready yet
        return false;
      }
      return true;
    };

    if (!addBloom()) {
      const id = setInterval(() => {
        if (addBloom()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }
  }, [renderMode]);

  // ── Ambient neurotransmitter particles ──
  const particlesRef = useRef<THREE.Points | null>(null);

  useEffect(() => {
    if (renderMode !== "3d") return;

    const addParticles = () => {
      const fg = fgRef.current as {
        scene?: () => THREE.Scene;
      } | null;
      if (!fg?.scene) return false;

      const scene = fg.scene();
      if (particlesRef.current) return true;

      const count = 200;
      const positions = new Float32Array(count * 3);
      const sizes = new Float32Array(count);

      for (let i = 0; i < count; i++) {
        // Random positions in a large sphere
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 50 + Math.random() * 450;
        positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
        positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3 + 2] = r * Math.cos(phi);
        sizes[i] = 0.3 + Math.random() * 0.5;
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3),
      );

      const material = new THREE.PointsMaterial({
        size: 0.5,
        color: 0x67e8f9,
        transparent: true,
        opacity: 0.15,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
        sizeAttenuation: true,
      });

      const particles = new THREE.Points(geometry, material);
      particles.userData.driftAxes = Array.from({ length: count }, () =>
        new THREE.Vector3(
          Math.random() - 0.5,
          Math.random() - 0.5,
          Math.random() - 0.5,
        ).normalize(),
      );
      scene.add(particles);
      particlesRef.current = particles;
      return true;
    };

    if (!addParticles()) {
      const id = setInterval(() => {
        if (addParticles()) clearInterval(id);
      }, 200);
      return () => clearInterval(id);
    }

    return () => {
      if (particlesRef.current) {
        const fg = fgRef.current as {
          scene?: () => THREE.Scene;
        } | null;
        if (fg?.scene) {
          fg.scene().remove(particlesRef.current);
        }
        particlesRef.current.geometry.dispose();
        (particlesRef.current.material as THREE.Material).dispose();
        particlesRef.current = null;
      }
    };
  }, [renderMode]);

  // ── Neural pulse animation loop (with cascade & breathing) ──
  const dirtyNodesRef = useRef<Set<string>>(new Set());
  const emittedPulsesRef = useRef<Set<string>>(new Set());
  const cascadeQueueRef = useRef<Map<string, CascadeEntry>>(new Map());
  const cascadedPulsesRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (renderMode !== "3d") return;

    let rafId: number;
    const startTime = Date.now();

    const animate = () => {
      rafId = requestAnimationFrame(animate);

      const pulses = useEngramStore.getState().activationPulses;
      const now = Date.now();
      const time = now - startTime;

      const fg = fgRef.current as {
        graphData?: () => {
          nodes: Array<Record<string, unknown>>;
          links: Array<Record<string, unknown>>;
        };
        emitParticle?: (link: unknown) => void;
      } | null;

      // ── Particle drift animation ──
      if (particlesRef.current) {
        const positions = particlesRef.current.geometry.attributes.position;
        const posArray = positions.array as Float32Array;
        const axes = particlesRef.current.userData.driftAxes as THREE.Vector3[];
        const rotSpeed = 0.0001;

        for (let i = 0; i < axes.length; i++) {
          const x = posArray[i * 3];
          const y = posArray[i * 3 + 1];
          const z = posArray[i * 3 + 2];
          const pos = new THREE.Vector3(x, y, z);
          pos.applyAxisAngle(axes[i], rotSpeed);
          posArray[i * 3] = pos.x;
          posArray[i * 3 + 1] = pos.y;
          posArray[i * 3 + 2] = pos.z;
        }
        positions.needsUpdate = true;
      }

      // Short-circuit further work if nothing is pulsing and nothing to clean up
      if (
        pulses.length === 0 &&
        dirtyNodesRef.current.size === 0 &&
        cascadeQueueRef.current.size === 0
      ) {
        // Still do breathing animation on existing nodes
        if (fg?.graphData) {
          const gd = fg.graphData();
          for (let idx = 0; idx < gd.nodes.length; idx++) {
            const graphNode = gd.nodes[idx];
            const threeObj = graphNode.__threeObj as THREE.Group | undefined;
            if (!threeObj) continue;

            threeObj.traverse((child) => {
              if (child instanceof THREE.Sprite) {
                const baseScale =
                  child.userData.baseScale ?? child.scale.x;
                if (!child.userData.baseScale)
                  child.userData.baseScale = baseScale;
                const breath =
                  1 + 0.05 * Math.sin(time * 0.001 + idx * 0.7);
                child.scale.setScalar(baseScale * breath);
              }
            });
          }
        }
        return;
      }

      if (!fg?.graphData) return;
      const gd = fg.graphData();
      const activePulseIds = new Set<string>();

      // ── Process cascade queue — fire secondary pulses ──
      for (const [key, entry] of cascadeQueueRef.current) {
        if (now >= entry.fireAt) {
          cascadeQueueRef.current.delete(key);
          // Create a cascade pulse
          useEngramStore.getState().addActivationPulse({
            entityId: entry.targetId,
            name: "",
            entityType: "",
            activation: entry.intensity,
            accessedVia: "cascade",
          });
        }
      }

      for (const pulse of pulses) {
        const intensity =
          pulse.cascadeIntensity !== undefined ? pulse.cascadeIntensity : 1;
        const elapsed = now - pulse.timestamp;
        if (elapsed > PULSE_DURATION_MS) continue;

        activePulseIds.add(pulse.entityId);
        dirtyNodesRef.current.add(pulse.entityId);

        // Quadratic ease-out: t goes 0→1, progress starts fast and slows
        const t = elapsed / PULSE_DURATION_MS;
        const easeOut = 1 - (1 - t) * (1 - t);
        const fadeIntensity = (1 - easeOut) * intensity;

        const graphNode = gd.nodes.find(
          (n: Record<string, unknown>) => n.id === pulse.entityId,
        );
        if (!graphNode) continue;

        const threeObj = graphNode.__threeObj as THREE.Group | undefined;
        if (!threeObj) continue;

        // Find sprite (glow) and soma mesh
        let sprite: THREE.Sprite | undefined;
        let somaMesh: THREE.Mesh | undefined;
        threeObj.traverse((child) => {
          if (child instanceof THREE.Sprite) sprite = child;
          else if (child instanceof THREE.Mesh) somaMesh = child;
        });

        if (sprite) {
          const baseScale = sprite.userData.baseScale ?? sprite.scale.x;
          sprite.userData.baseScale = baseScale;
          const scaleMultiplier = 1 + 1.5 * fadeIntensity;
          sprite.scale.setScalar(baseScale * scaleMultiplier);

          const baseMat = sprite.material as THREE.SpriteMaterial;
          const baseOpacity = baseMat.userData?.baseOpacity ?? baseMat.opacity;
          baseMat.userData = { baseOpacity };
          baseMat.opacity = baseOpacity + 0.5 * fadeIntensity;
        }

        if (somaMesh) {
          const mat = somaMesh.material as THREE.MeshStandardMaterial;
          const baseEmissiveIntensity =
            mat.userData?.baseEmissiveIntensity ?? mat.emissiveIntensity;
          if (mat.userData?.baseEmissiveIntensity === undefined) {
            mat.userData = {
              ...mat.userData,
              baseEmissiveIntensity: mat.emissiveIntensity,
              baseColor: mat.emissive.getHex(),
            };
          }
          // Boost emissive toward white during pulse
          const white = new THREE.Color(0xffffff);
          const baseEmissive = new THREE.Color(mat.userData.baseColor);
          mat.emissive.copy(baseEmissive).lerp(white, 0.7 * fadeIntensity);
          mat.emissiveIntensity =
            baseEmissiveIntensity + 2.0 * fadeIntensity;
        }

        // Emit particles along connected edges (once per pulse)
        const pulseKey = `${pulse.entityId}-${pulse.timestamp}`;
        if (!emittedPulsesRef.current.has(pulseKey) && fg.emitParticle) {
          emittedPulsesRef.current.add(pulseKey);
          for (const link of gd.links) {
            const src =
              typeof link.source === "object"
                ? (link.source as Record<string, unknown>).id
                : link.source;
            const tgt =
              typeof link.target === "object"
                ? (link.target as Record<string, unknown>).id
                : link.target;
            if (src === pulse.entityId || tgt === pulse.entityId) {
              fg.emitParticle(link);
            }
          }

          // Queue cascade to neighbors (1-hop only, no re-triggering)
          if (
            pulse.accessedVia !== "cascade" &&
            !cascadedPulsesRef.current.has(pulseKey) &&
            cascadeQueueRef.current.size < CASCADE_MAX_QUEUE
          ) {
            cascadedPulsesRef.current.add(pulseKey);
            for (const link of gd.links) {
              const src =
                typeof link.source === "object"
                  ? (link.source as Record<string, unknown>).id
                  : link.source;
              const tgt =
                typeof link.target === "object"
                  ? (link.target as Record<string, unknown>).id
                  : link.target;

              let neighborId: string | null = null;
              if (src === pulse.entityId)
                neighborId = tgt as string;
              else if (tgt === pulse.entityId)
                neighborId = src as string;

              if (
                neighborId &&
                cascadeQueueRef.current.size < CASCADE_MAX_QUEUE
              ) {
                const cascadeKey = `${neighborId}-${pulse.timestamp}`;
                if (!cascadeQueueRef.current.has(cascadeKey)) {
                  cascadeQueueRef.current.set(cascadeKey, {
                    targetId: neighborId,
                    fireAt: now + CASCADE_DELAY_MS,
                    intensity: CASCADE_INTENSITY,
                  });
                }
              }
            }
          }
        }
      }

      // Reset dirty nodes no longer pulsing
      for (const nodeId of dirtyNodesRef.current) {
        if (activePulseIds.has(nodeId)) continue;

        const graphNode = gd.nodes.find(
          (n: Record<string, unknown>) => n.id === nodeId,
        );
        if (!graphNode) {
          dirtyNodesRef.current.delete(nodeId);
          continue;
        }

        const threeObj = graphNode.__threeObj as THREE.Group | undefined;
        if (!threeObj) {
          dirtyNodesRef.current.delete(nodeId);
          continue;
        }

        threeObj.traverse((child) => {
          if (child instanceof THREE.Sprite) {
            const baseScale = child.userData.baseScale;
            if (baseScale != null) child.scale.setScalar(baseScale);
            const baseMat = child.material as THREE.SpriteMaterial;
            if (baseMat.userData?.baseOpacity != null) {
              baseMat.opacity = baseMat.userData.baseOpacity;
            }
          } else if (child instanceof THREE.Mesh) {
            const mat = child.material as THREE.MeshStandardMaterial;
            if (mat.userData?.baseColor != null) {
              mat.emissive.setHex(mat.userData.baseColor);
            }
            if (mat.userData?.baseEmissiveIntensity != null) {
              mat.emissiveIntensity = mat.userData.baseEmissiveIntensity;
            }
          }
        });

        dirtyNodesRef.current.delete(nodeId);
      }

      // ── Breathing animation — subtle sine-wave on all glow sprites ──
      for (let idx = 0; idx < gd.nodes.length; idx++) {
        if (activePulseIds.has(gd.nodes[idx].id as string)) continue;

        const graphNode = gd.nodes[idx];
        const threeObj = graphNode.__threeObj as THREE.Group | undefined;
        if (!threeObj) continue;

        threeObj.traverse((child) => {
          if (child instanceof THREE.Sprite) {
            const baseScale = child.userData.baseScale ?? child.scale.x;
            if (!child.userData.baseScale)
              child.userData.baseScale = baseScale;
            const breath = 1 + 0.05 * Math.sin(time * 0.001 + idx * 0.7);
            child.scale.setScalar(baseScale * breath);
          }
        });
      }

      // Clean up old emitted keys
      if (emittedPulsesRef.current.size > 200) {
        emittedPulsesRef.current.clear();
      }
      if (cascadedPulsesRef.current.size > 200) {
        cascadedPulsesRef.current.clear();
      }
    };

    rafId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafId);
  }, [renderMode]);

  const handleNodeClick = useCallback(
    (node: { id?: string }) => selectNode(node.id ?? null),
    [selectNode],
  );
  const handleNodeRightClick = useCallback(
    (node: { id?: string }) => {
      if (node.id) expandNode(node.id);
    },
    [expandNode],
  );
  const handleNodeHover = useCallback(
    (node: { id?: string } | null) => hoverNode(node?.id ?? null),
    [hoverNode],
  );

  const nodeThreeObject = useCallback(
    (node: Record<string, unknown>) => createNodeObject(node, showHeatmap),
    [showHeatmap],
  );

  if (!isLoading && graphData.nodes.length === 0) return <EmptyState />;

  const is3D = renderMode === "3d";

  return (
    <div className="relative h-full w-full">
      {isLoading && (
        <div
          className="absolute inset-0 z-10 flex items-center justify-center"
          style={{ background: "rgba(10, 10, 26, 0.6)" }}
        >
          <div
            className="animate-pulse-soft"
            style={{
              fontFamily: "var(--font-display)",
              fontSize: 18,
              color: "var(--accent)",
              fontStyle: "italic",
            }}
          >
            Loading memories...
          </div>
        </div>
      )}
      {is3D ? (
        <ForceGraph3D
          ref={fgRef as React.MutableRefObject<never>}
          graphData={graphData}
          nodeId="id"
          nodeLabel="name"
          nodeThreeObject={nodeThreeObject}
          linkSource="source"
          linkTarget="target"
          linkCurvature={0.25}
          linkCurveRotation={(link: Record<string, unknown>) =>
            (link.__curveRotation as number) ?? 0
          }
          linkResolution={12}
          linkWidth={(link: Record<string, unknown>) =>
            Math.max(0.3, ((link.weight as number) ?? 1) * 1.2)
          }
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(120, 160, 255, 0.3)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={2.0}
          linkDirectionalParticleColor={() => "#c4b5fd"}
          linkDirectionalParticleSpeed={0.002}
          linkOpacity={0.4}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onNodeHover={handleNodeHover}
          backgroundColor="#0a0a1a"
          showNavInfo={false}
        />
      ) : (
        <ForceGraph2D
          graphData={graphData}
          nodeId="id"
          nodeLabel="name"
          nodeColor={(n: Record<string, unknown>) =>
            showHeatmap
              ? activationColor((n.activationCurrent as number) ?? 0)
              : entityColor((n.entityType as string) ?? "Other")
          }
          nodeVal={(n: Record<string, unknown>) =>
            2 + ((n.activationCurrent as number) ?? 0) * 12
          }
          linkSource="source"
          linkTarget="target"
          linkWidth={(link: Record<string, unknown>) =>
            Math.max(0.5, ((link.weight as number) ?? 1) * 1.5)
          }
          linkLabel={
            showEdgeLabels
              ? (link: Record<string, unknown>) => link.predicate as string
              : undefined
          }
          linkColor={() => "rgba(120, 160, 255, 0.35)"}
          linkDirectionalParticles={1}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleColor={() => "#c4b5fd"}
          onNodeClick={handleNodeClick}
          onNodeRightClick={handleNodeRightClick}
          onNodeHover={handleNodeHover}
          backgroundColor="#0a0a1a"
        />
      )}
      <NodeTooltip />
    </div>
  );
}
