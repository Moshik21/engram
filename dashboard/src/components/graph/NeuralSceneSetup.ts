import * as THREE from "three";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";

type FgRef = {
  renderer?: () => THREE.WebGLRenderer;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  postProcessingComposer?: () => any;
  scene?: () => THREE.Scene;
} | null;

/**
 * Add UnrealBloomPass to a ForceGraph3D instance.
 * Polls until the renderer is available, then adds the pass once.
 * Returns a cleanup function.
 */
export function setupBloomPass(fgRef: React.RefObject<FgRef>): () => void {
  let added = false;
  let intervalId: ReturnType<typeof setInterval> | null = null;

  const addBloom = () => {
    const fg = fgRef.current;
    if (!fg?.renderer || added) return !!added;

    try {
      const renderer = fg.renderer();
      const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(
          renderer.domElement.width * 0.5,
          renderer.domElement.height * 0.5,
        ),
        0.4, // strength
        0.4, // radius
        0.85, // threshold
      );

      if (fg.postProcessingComposer) {
        fg.postProcessingComposer().addPass(bloomPass);
      }

      added = true;
    } catch {
      return false;
    }
    return true;
  };

  if (!addBloom()) {
    intervalId = setInterval(() => {
      if (addBloom() && intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }, 200);
  }

  return () => {
    if (intervalId) clearInterval(intervalId);
  };
}

export interface AmbientParticles {
  points: THREE.Points;
  animate(): void;
  cleanup(): void;
}

/**
 * Create drifting ambient neurotransmitter particles and add them to the scene.
 * Returns an object with an animate() callback (call each frame) and cleanup().
 */
export function createAmbientParticles(
  fgRef: React.RefObject<FgRef>,
  count: number,
  radius: number,
): { result: AmbientParticles | null; cleanup: () => void } {
  let particles: THREE.Points | null = null;
  let scene: THREE.Scene | null = null;
  let intervalId: ReturnType<typeof setInterval> | null = null;
  let resultRef: AmbientParticles | null = null;

  const addParticles = () => {
    const fg = fgRef.current;
    if (!fg?.scene || particles) return !!particles;

    scene = fg.scene();
    const positions = new Float32Array(count * 3);

    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 50 + Math.random() * radius;
      positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      positions[i * 3 + 2] = r * Math.cos(phi);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      size: 0.5,
      color: 0x67e8f9,
      transparent: true,
      opacity: 0.2,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      sizeAttenuation: true,
    });

    particles = new THREE.Points(geometry, material);
    particles.userData.driftAxes = Array.from({ length: count }, () =>
      new THREE.Vector3(
        Math.random() - 0.5,
        Math.random() - 0.5,
        Math.random() - 0.5,
      ).normalize(),
    );
    scene.add(particles);

    resultRef = {
      points: particles,
      animate: () => {
        if (!particles) return;
        const posAttr = particles.geometry.attributes.position;
        const posArray = posAttr.array as Float32Array;
        const axes = particles.userData.driftAxes as THREE.Vector3[];
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
        posAttr.needsUpdate = true;
      },
      cleanup: () => {
        if (particles && scene) {
          scene.remove(particles);
          particles.geometry.dispose();
          (particles.material as THREE.Material).dispose();
          particles = null;
        }
      },
    };

    return true;
  };

  if (!addParticles()) {
    intervalId = setInterval(() => {
      if (addParticles() && intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    }, 200);
  }

  return {
    get result() {
      return resultRef;
    },
    cleanup: () => {
      if (intervalId) clearInterval(intervalId);
      resultRef?.cleanup();
    },
  };
}
