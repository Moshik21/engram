// @ts-nocheck — R3F JSX intrinsic elements are resolved at runtime by the Canvas
import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import {
  ACESFilmicToneMapping,
  AdditiveBlending,
  BufferAttribute,
  Color,
  Group,
  InstancedMesh,
  LineSegments,
  Object3D,
  Points,
  Vector3,
} from "three";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const NODE_COUNT = 68;
const EDGE_COUNT = 105;
const PARTICLE_COUNT = 200;
const CLUSTER_COUNT = 4;

const NODE_COLORS: string[] = [
  "#67e8f9", // cyan
  "#a78bfa", // purple
  "#34d399", // green
  "#f97316", // orange
  "#818cf8", // indigo
];

const NODE_COLOR_OBJECTS = NODE_COLORS.map((c) => new Color(c));

const EDGE_COLOR = new Color("#67e8f9");

// ---------------------------------------------------------------------------
// Layout helpers – golden-ratio spherical clusters
// ---------------------------------------------------------------------------

interface NodeData {
  position: Vector3;
  color: Color;
  radius: number;
  phase: number; // animation phase offset
  pulseSpeed: number;
  pulseDelay: number; // seconds before first pulse
  cluster: number;
}

interface EdgeData {
  a: number;
  b: number;
  weight: number; // 0-1
}

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function generateNodes(count: number): NodeData[] {
  const rand = seededRandom(42);
  const nodes: NodeData[] = [];

  // Cluster centres spread in a sphere
  const clusterCentres: Vector3[] = [];
  for (let c = 0; c < CLUSTER_COUNT; c++) {
    const phi = Math.acos(1 - (2 * (c + 0.5)) / CLUSTER_COUNT);
    const theta = Math.PI * (1 + Math.sqrt(5)) * c;
    const r = 2.8 + rand() * 0.6;
    clusterCentres.push(
      new Vector3(
        r * Math.sin(phi) * Math.cos(theta),
        r * Math.sin(phi) * Math.sin(theta),
        r * Math.cos(phi),
      ),
    );
  }

  for (let i = 0; i < count; i++) {
    // Assign to a cluster with a bias toward filling evenly
    const cluster = i % CLUSTER_COUNT;
    const centre = clusterCentres[cluster];

    // Golden-ratio spiral within the cluster + jitter
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const t = i / count;
    const inclination = Math.acos(1 - 2 * t);
    const azimuth = goldenAngle * i;

    const spread = 1.8 + rand() * 1.2;
    const jitter = new Vector3(
      (rand() - 0.5) * 1.0,
      (rand() - 0.5) * 1.0,
      (rand() - 0.5) * 1.0,
    );

    const pos = new Vector3(
      Math.sin(inclination) * Math.cos(azimuth) * spread,
      Math.sin(inclination) * Math.sin(azimuth) * spread,
      Math.cos(inclination) * spread,
    )
      .add(centre.clone().multiplyScalar(0.45))
      .add(jitter);

    const colorIdx = Math.floor(rand() * NODE_COLOR_OBJECTS.length);
    const radius = 0.04 + rand() * 0.08;

    nodes.push({
      position: pos,
      color: NODE_COLOR_OBJECTS[colorIdx],
      radius,
      phase: rand() * Math.PI * 2,
      pulseSpeed: 0.3 + rand() * 0.7,
      pulseDelay: rand() * 12,
      cluster,
    });
  }

  return nodes;
}

function generateEdges(
  nodes: NodeData[],
  count: number,
): EdgeData[] {
  const rand = seededRandom(99);
  const edges: EdgeData[] = [];
  const seen = new Set<string>();

  // Prefer intra-cluster edges (70 %) then inter-cluster (30 %)
  const intraTarget = Math.floor(count * 0.7);

  // Build cluster index
  const clusterMap = new Map<number, number[]>();
  nodes.forEach((n, i) => {
    const arr = clusterMap.get(n.cluster) ?? [];
    arr.push(i);
    clusterMap.set(n.cluster, arr);
  });
  const clusterIds = [...clusterMap.keys()];

  const addEdge = (a: number, b: number) => {
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (seen.has(key) || a === b) return false;
    seen.add(key);
    edges.push({ a, b, weight: 0.15 + rand() * 0.85 });
    return true;
  };

  // Intra-cluster
  let attempts = 0;
  while (edges.length < intraTarget && attempts < count * 8) {
    attempts++;
    const cid = clusterIds[Math.floor(rand() * clusterIds.length)];
    const members = clusterMap.get(cid)!;
    if (members.length < 2) continue;
    const ai = members[Math.floor(rand() * members.length)];
    const bi = members[Math.floor(rand() * members.length)];
    addEdge(ai, bi);
  }

  // Inter-cluster
  attempts = 0;
  while (edges.length < count && attempts < count * 8) {
    attempts++;
    const ai = Math.floor(rand() * nodes.length);
    let bi = Math.floor(rand() * nodes.length);
    // Ensure different cluster
    if (nodes[ai].cluster === nodes[bi].cluster) {
      bi = (bi + Math.floor(nodes.length / 2)) % nodes.length;
    }
    addEdge(ai, bi);
  }

  return edges;
}

// ---------------------------------------------------------------------------
// Instanced Nodes
// ---------------------------------------------------------------------------

interface NodesProps {
  nodes: NodeData[];
}

const _tempObj = new Object3D();
const _tempColor = new Color();

function Nodes({ nodes }: NodesProps) {
  const meshRef = useRef<InstancedMesh>(null!);
  const count = nodes.length;

  // Store per-instance base emissive intensity for pulsing
  const baseEmissive = useMemo(
    () => nodes.map(() => 0.35),
    [nodes],
  );

  useFrame(({ clock }) => {
    const mesh = meshRef.current;
    if (!mesh) return;

    const t = clock.getElapsedTime();

    for (let i = 0; i < count; i++) {
      const node = nodes[i];

      // Floating sine oscillation
      const floatY = Math.sin(t * 0.4 + node.phase) * 0.06;
      const floatX = Math.cos(t * 0.3 + node.phase * 1.3) * 0.03;

      _tempObj.position.set(
        node.position.x + floatX,
        node.position.y + floatY,
        node.position.z,
      );

      // Pulse: periodic brightness spike
      const pulseTime = (t - node.pulseDelay) * node.pulseSpeed;
      const pulseCycle = pulseTime % 8; // every ~8s adjusted by speed
      const pulseIntensity =
        pulseCycle > 0 && pulseCycle < 0.6
          ? Math.sin((pulseCycle / 0.6) * Math.PI) * 0.65
          : 0;

      const scale = node.radius * (1 + pulseIntensity * 0.3);
      _tempObj.scale.setScalar(scale);
      _tempObj.updateMatrix();
      mesh.setMatrixAt(i, _tempObj.matrix);

      // Color with emissive boost for pulse
      const emissiveStr = baseEmissive[i] + pulseIntensity;
      _tempColor.copy(node.color).multiplyScalar(0.4 + emissiveStr * 0.6);
      mesh.setColorAt(i, _tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 12, 8]} />
      <meshStandardMaterial
        toneMapped={false}
        emissive="#ffffff"
        emissiveIntensity={0.5}
        roughness={0.4}
        metalness={0.1}
      />
    </instancedMesh>
  );
}

// ---------------------------------------------------------------------------
// Edges (LineSegments)
// ---------------------------------------------------------------------------

interface EdgesProps {
  nodes: NodeData[];
  edges: EdgeData[];
}

function Edges({ nodes, edges }: EdgesProps) {
  const lineRef = useRef<LineSegments>(null!);

  const { positions, colors } = useMemo(() => {
    const pos = new Float32Array(edges.length * 6);
    const col = new Float32Array(edges.length * 6);

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      const pa = nodes[e.a].position;
      const pb = nodes[e.b].position;
      const off = i * 6;

      pos[off] = pa.x;
      pos[off + 1] = pa.y;
      pos[off + 2] = pa.z;
      pos[off + 3] = pb.x;
      pos[off + 4] = pb.y;
      pos[off + 5] = pb.z;

      // Color with alpha baked into brightness
      const alpha = 0.04 + e.weight * 0.10;
      col[off] = EDGE_COLOR.r * alpha;
      col[off + 1] = EDGE_COLOR.g * alpha;
      col[off + 2] = EDGE_COLOR.b * alpha;
      col[off + 3] = EDGE_COLOR.r * alpha;
      col[off + 4] = EDGE_COLOR.g * alpha;
      col[off + 5] = EDGE_COLOR.b * alpha;
    }

    return { positions: pos, colors: col };
  }, [nodes, edges]);

  // Animate edge positions to follow floating nodes
  useFrame(({ clock }) => {
    const line = lineRef.current;
    if (!line) return;

    const geo = line.geometry;
    const posAttr = geo.getAttribute("position") as BufferAttribute;
    const posArr = posAttr.array as Float32Array;
    const t = clock.getElapsedTime();

    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      const na = nodes[e.a];
      const nb = nodes[e.b];
      const off = i * 6;

      const floatYa = Math.sin(t * 0.4 + na.phase) * 0.06;
      const floatXa = Math.cos(t * 0.3 + na.phase * 1.3) * 0.03;
      posArr[off] = na.position.x + floatXa;
      posArr[off + 1] = na.position.y + floatYa;
      posArr[off + 2] = na.position.z;

      const floatYb = Math.sin(t * 0.4 + nb.phase) * 0.06;
      const floatXb = Math.cos(t * 0.3 + nb.phase * 1.3) * 0.03;
      posArr[off + 3] = nb.position.x + floatXb;
      posArr[off + 4] = nb.position.y + floatYb;
      posArr[off + 5] = nb.position.z;
    }

    posAttr.needsUpdate = true;
  });

  return (
    <lineSegments ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
        <bufferAttribute
          attach="attributes-color"
          args={[colors, 3]}
        />
      </bufferGeometry>
      <lineBasicMaterial
        vertexColors
        transparent
        opacity={1}
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </lineSegments>
  );
}

// ---------------------------------------------------------------------------
// Floating Particles
// ---------------------------------------------------------------------------

function Particles() {
  const pointsRef = useRef<Points>(null!);

  const { positions, velocities } = useMemo(() => {
    const rand = seededRandom(777);
    const pos = new Float32Array(PARTICLE_COUNT * 3);
    const vel = new Float32Array(PARTICLE_COUNT * 3);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const off = i * 3;
      // Distribute in a large sphere
      const theta = rand() * Math.PI * 2;
      const phi = Math.acos(2 * rand() - 1);
      const r = 1.5 + rand() * 5.0;
      pos[off] = r * Math.sin(phi) * Math.cos(theta);
      pos[off + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[off + 2] = r * Math.cos(phi);

      // Slow drift velocity
      vel[off] = (rand() - 0.5) * 0.008;
      vel[off + 1] = (rand() - 0.5) * 0.008;
      vel[off + 2] = (rand() - 0.5) * 0.008;
    }

    return { positions: pos, velocities: vel };
  }, []);

  useFrame(() => {
    const pts = pointsRef.current;
    if (!pts) return;

    const posAttr = pts.geometry.getAttribute(
      "position",
    ) as BufferAttribute;
    const arr = posAttr.array as Float32Array;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const off = i * 3;
      arr[off] += velocities[off];
      arr[off + 1] += velocities[off + 1];
      arr[off + 2] += velocities[off + 2];

      // Wrap back if too far from origin
      const dist = Math.sqrt(
        arr[off] ** 2 + arr[off + 1] ** 2 + arr[off + 2] ** 2,
      );
      if (dist > 7) {
        arr[off] *= -0.3;
        arr[off + 1] *= -0.3;
        arr[off + 2] *= -0.3;
      }
    }

    posAttr.needsUpdate = true;
  });

  return (
    <points ref={pointsRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.025}
        color="#ffffff"
        transparent
        opacity={0.25}
        depthWrite={false}
        blending={AdditiveBlending}
        sizeAttenuation
      />
    </points>
  );
}

// ---------------------------------------------------------------------------
// Glow Cores — large faint additive spheres behind key nodes for bloom feel
// ---------------------------------------------------------------------------

interface GlowCoresProps {
  nodes: NodeData[];
}

function GlowCores({ nodes }: GlowCoresProps) {
  // Pick ~15 larger nodes to get a subtle glow halo
  const glowNodes = useMemo(() => {
    return nodes
      .map((n, i) => ({ ...n, idx: i }))
      .sort((a, b) => b.radius - a.radius)
      .slice(0, 15);
  }, [nodes]);

  const meshRef = useRef<InstancedMesh>(null!);
  const count = glowNodes.length;

  useFrame(({ clock }) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const t = clock.getElapsedTime();

    for (let i = 0; i < count; i++) {
      const node = glowNodes[i];
      const floatY = Math.sin(t * 0.4 + node.phase) * 0.06;
      const floatX = Math.cos(t * 0.3 + node.phase * 1.3) * 0.03;

      _tempObj.position.set(
        node.position.x + floatX,
        node.position.y + floatY,
        node.position.z,
      );
      _tempObj.scale.setScalar(node.radius * 4);
      _tempObj.updateMatrix();
      mesh.setMatrixAt(i, _tempObj.matrix);

      _tempColor.copy(node.color).multiplyScalar(0.15);
      mesh.setColorAt(i, _tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 8, 6]} />
      <meshBasicMaterial
        transparent
        opacity={0.12}
        depthWrite={false}
        blending={AdditiveBlending}
        toneMapped={false}
      />
    </instancedMesh>
  );
}

// ---------------------------------------------------------------------------
// Activation Pulses — travelling light along random edges
// ---------------------------------------------------------------------------

interface PulseTravellersProps {
  nodes: NodeData[];
  edges: EdgeData[];
}

function PulseTravellers({ nodes, edges }: PulseTravellersProps) {
  const PULSE_COUNT = 8;
  const meshRef = useRef<InstancedMesh>(null!);

  // Pick a random subset of edges to pulse along, cycling
  const pulseData = useMemo(() => {
    const rand = seededRandom(333);
    return Array.from({ length: PULSE_COUNT }, () => ({
      edgeIdx: Math.floor(rand() * edges.length),
      speed: 0.15 + rand() * 0.25,
      offset: rand() * 20,
      colorIdx: Math.floor(rand() * NODE_COLOR_OBJECTS.length),
    }));
  }, [edges]);

  useFrame(({ clock }) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const t = clock.getElapsedTime();

    for (let i = 0; i < PULSE_COUNT; i++) {
      const pd = pulseData[i];
      const edge = edges[pd.edgeIdx];
      const na = nodes[edge.a];
      const nb = nodes[edge.b];

      // Parameter along edge, ping-pong
      const raw = ((t + pd.offset) * pd.speed) % 2;
      const param = raw > 1 ? 2 - raw : raw;

      const floatYa = Math.sin(t * 0.4 + na.phase) * 0.06;
      const floatXa = Math.cos(t * 0.3 + na.phase * 1.3) * 0.03;
      const floatYb = Math.sin(t * 0.4 + nb.phase) * 0.06;
      const floatXb = Math.cos(t * 0.3 + nb.phase * 1.3) * 0.03;

      const ax = na.position.x + floatXa;
      const ay = na.position.y + floatYa;
      const az = na.position.z;
      const bx = nb.position.x + floatXb;
      const by = nb.position.y + floatYb;
      const bz = nb.position.z;

      _tempObj.position.set(
        ax + (bx - ax) * param,
        ay + (by - ay) * param,
        az + (bz - az) * param,
      );
      _tempObj.scale.setScalar(0.025);
      _tempObj.updateMatrix();
      mesh.setMatrixAt(i, _tempObj.matrix);

      _tempColor.copy(NODE_COLOR_OBJECTS[pd.colorIdx]);
      mesh.setColorAt(i, _tempColor);
    }

    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, PULSE_COUNT]}
    >
      <sphereGeometry args={[1, 6, 4]} />
      <meshBasicMaterial
        toneMapped={false}
        transparent
        opacity={0.9}
        depthWrite={false}
        blending={AdditiveBlending}
      />
    </instancedMesh>
  );
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

function BrainScene() {
  const nodes = useMemo(() => generateNodes(NODE_COUNT), []);
  const edges = useMemo(() => generateEdges(nodes, EDGE_COUNT), [nodes]);
  const sceneRef = useRef<Group>(null!);

  useFrame(({ clock }) => {
    const group = sceneRef.current;
    if (!group) return;
    const t = clock.getElapsedTime();
    group.rotation.y = t * 0.05;
    group.rotation.x = Math.sin(t * 0.12) * 0.05;
  });

  return (
    <>
      <ambientLight intensity={0.15} />
      <pointLight position={[10, 10, 10]} intensity={0.3} />
      <pointLight position={[-10, -5, -10]} intensity={0.15} color="#a78bfa" />

      <group ref={sceneRef}>
        <Nodes nodes={nodes} />
        <Edges nodes={nodes} edges={edges} />
        <GlowCores nodes={nodes} />
        <Particles />
        <PulseTravellers nodes={nodes} edges={edges} />
      </group>
    </>
  );
}

// ---------------------------------------------------------------------------
// Exported: Full-screen background component
// ---------------------------------------------------------------------------

export function BrainVisualization() {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 8], fov: 50, near: 0.1, far: 100 }}
        dpr={[1, 1.5]}
        gl={{
          antialias: true,
          alpha: true,
          powerPreference: "high-performance",
          toneMapping: ACESFilmicToneMapping,
          toneMappingExposure: 1.2,
        }}
        style={{ background: "transparent" }}
      >
        <BrainScene />
      </Canvas>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported: Hero wrapper with HTML overlay
// ---------------------------------------------------------------------------

interface BrainHeroProps {
  children?: React.ReactNode;
}

export function BrainHero({ children }: BrainHeroProps) {
  return (
    <section
      className="relative w-full overflow-hidden"
      style={{ minHeight: "100vh" }}
    >
      {/* 3D Background */}
      <div className="absolute inset-0 z-0">
        <BrainVisualization />
      </div>

      {/* Radial gradient overlay for depth & readability */}
      <div
        className="pointer-events-none absolute inset-0 z-[1]"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 30%, rgba(0,0,0,0.55) 100%)",
        }}
      />

      {/* Content overlay */}
      <div className="relative z-[2] flex min-h-screen items-center justify-center">
        <div className="mx-auto max-w-4xl px-6 text-center">{children}</div>
      </div>
    </section>
  );
}
