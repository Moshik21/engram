import type { GraphNode, GraphEdge } from "../store/types";

const ENTITY_TYPES = [
  "Person", "Organization", "Technology", "Location", "Concept",
  "Event", "Project", "Skill", "Tool", "Language",
  "Framework", "Service", "Topic", "Goal", "Preference",
  "Workflow", "Other",
];

export function generateSyntheticGraph(
  nodeCount: number,
  edgesPerNode = 2.5,
): { nodes: Record<string, GraphNode>; edges: Record<string, GraphEdge> } {
  const nodes: Record<string, GraphNode> = {};
  const edges: Record<string, GraphEdge> = {};
  const now = new Date().toISOString();

  for (let i = 0; i < nodeCount; i++) {
    const id = `synth-node-${i}`;
    nodes[id] = {
      id,
      name: `Node ${i}`,
      entityType: ENTITY_TYPES[i % ENTITY_TYPES.length],
      summary: null,
      activationCurrent: Math.random(),
      accessCount: Math.floor(Math.random() * 50),
      lastAccessed: now,
      createdAt: now,
      updatedAt: now,
    };
  }

  const totalEdges = Math.floor(nodeCount * edgesPerNode);
  const nodeIds = Object.keys(nodes);

  for (let i = 0; i < totalEdges; i++) {
    const srcIdx = Math.floor(Math.random() * nodeCount);
    let tgtIdx = Math.floor(Math.random() * nodeCount);
    if (tgtIdx === srcIdx) tgtIdx = (srcIdx + 1) % nodeCount;

    const id = `synth-edge-${i}`;
    edges[id] = {
      id,
      source: nodeIds[srcIdx],
      target: nodeIds[tgtIdx],
      predicate: "RELATED_TO",
      weight: 0.3 + Math.random() * 0.7,
      validFrom: null,
      validTo: null,
      createdAt: now,
    };
  }

  return { nodes, edges };
}
