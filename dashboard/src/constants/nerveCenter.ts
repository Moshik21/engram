import type { NeuralSpecialization } from "../store/types";
import type { ConsolidationPhaseName } from "./consolidation";

export const NEURAL_SPECIALIZATIONS: Record<NeuralSpecialization, { icon: string; color: string; description: string }> = {
  Architect: { icon: "🏗️", color: "#60a5fa", description: "Master of system architecture and infrastructure" },
  Synthesizer: { icon: "🧠", color: "#818cf8", description: "Expert in synaptic synthesis and conceptual mapping" },
  Narrator: { icon: "🕸️", color: "#f472b6", description: "Weaver of narrative threads and creative output" },
  Integrator: { icon: "🤝", color: "#34d399", description: "Manager of interpersonal nodes and social links" },
  Biochemist: { icon: "🧪", color: "#fb923c", description: "Optimizer of somatic health and wellness metrics" },
  Topologist: { icon: "🗺️", color: "#a78bfa", description: "Explorer of spatial grids and organizational hierarchies" },
  Polymath: { icon: "⚛️", color: "#D4A84B", description: "Cognitive generalist across all neural domains" },
};

export const DOMAIN_TO_REGION: Record<string, string> = {
  technical: "The Logic Core",
  knowledge: "The Synaptic Archive",
  creative: "The Creative Cortex",
  personal: "The Social Interface",
  health: "The Somatic Buffer",
  spatial: "The Topographic Grid",
  general: "The Primary Ingress",
};

export const EVENT_FLAVOR_TEXT: Record<string, { text: string; plasticity: number }> = {
  "episode.completed": { text: "Signal successfully processed!", plasticity: 1 },
  "episode.queued": { text: "Inbound stimulus detected...", plasticity: 0 },
  "consolidation.phase.merge.completed": { text: "Duplicate patterns synchronized.", plasticity: 5 },
  "consolidation.phase.dream.completed": { text: "Deep inference complete: cross-domain links formed.", plasticity: 3 },
  "consolidation.phase.prune.completed": { text: "Noise filtered from the synaptic field.", plasticity: 2 },
  "consolidation.completed": { text: "Cycle complete. Neural coherence improved.", plasticity: 15 },
  "consolidation.started": { text: "Plasticity phase initiated...", plasticity: 0 },
  "feedback.recorded": { text: "Synaptic weighting updated.", plasticity: 2 },
};

export const NERVE_ACCENT = "#67e8f9";
export const NERVE_GLOW = "rgba(103, 232, 249, 0.15)";

export const ACHIEVEMENT_DEFINITIONS: Array<{
  id: string;
  name: string;
  description: string;
  icon: string;
  condition: (stats: { totalEntities: number; totalRelationships: number; totalEpisodes: number; level: number }) => boolean;
}> = [
  { id: "first_memory", name: "Inception", description: "First episode encoded", icon: "\u2726",
    condition: (s) => s.totalEpisodes >= 1 },
  { id: "knowledge_seeker", name: "Data Miner", description: "50 entities registered", icon: "\u2605",
    condition: (s) => s.totalEntities >= 50 },
  { id: "web_weaver", name: "Synapse Builder", description: "100 relationships formed", icon: "\u2726",
    condition: (s) => s.totalRelationships >= 100 },
  { id: "memory_palace", name: "Cerebral Citadel", description: "500 entities registered", icon: "\u265B",
    condition: (s) => s.totalEntities >= 500 },
  { id: "level_5", name: "Integrated", description: "Reach cortical level 5", icon: "🟢",
    condition: (s) => s.level >= 5 },
  { id: "level_10", name: "Advanced", description: "Reach cortical level 10", icon: "🔵",
    condition: (s) => s.level >= 10 },
  { id: "archivist", name: "Systems Analyst", description: "100 episodes encoded", icon: "📊",
    condition: (s) => s.totalEpisodes >= 100 },
  { id: "level_20", name: "Omni-Cognizant", description: "Reach cortical level 20", icon: "🟣",
    condition: (s) => s.level >= 20 },
];

export const PHASE_DESCRIPTIONS: Record<ConsolidationPhaseName, string> = {
  triage: "Scanning inbound stimulus for signal value",
  merge: "Synchronizing high-entropy duplicate nodes",
  calibrate: "Attuning neural weights to user feedback",
  infer: "Simulating latent connections across the field",
  evidence_adjudication: "Weighing contradictory evidence for final commit",
  edge_adjudication: "Evaluating uncertain links in the synaptic map",
  replay: "Reliving episodes for pattern recognition",
  prune: "Filtering low-gravity noise and weak connections",
  compact: "Compressing access history into efficient buffers",
  reflect: "Synthesizing durable observations from related memories",
  reindex: "Re-cataloging the neural storage indices",
  graph_embed: "Encoding structural geometry into vectors",
  microglia: "Cleansing corrupted summaries and orphaned nodes",
  immunity: "Dissolving low-semantic-gravity topological noise",
  dream: "Simulating cross-domain synaptic bridges",
};
