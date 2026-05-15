import type { PlayerClass } from "../store/types";
import type { ConsolidationPhaseName } from "./consolidation";

export const PLAYER_CLASSES: Record<PlayerClass, { icon: string; color: string; description: string }> = {
  Artificer: { icon: "\u2699", color: "#67e8f9", description: "Master of technology and systems" },
  Sage: { icon: "\u2726", color: "#818cf8", description: "Seeker of knowledge and concepts" },
  Bard: { icon: "\u266A", color: "#f472b6", description: "Creator of works and stories" },
  Diplomat: { icon: "\u2606", color: "#34d399", description: "Weaver of personal connections" },
  Alchemist: { icon: "\u2697", color: "#fb923c", description: "Guardian of health and wellness" },
  Cartographer: { icon: "\u25C9", color: "#a78bfa", description: "Explorer of places and organizations" },
  Polymath: { icon: "\u2726", color: "#D4A84B", description: "Renaissance mind across all domains" },
};

export const DOMAIN_TO_CONTINENT: Record<string, string> = {
  technical: "The Forge",
  knowledge: "The Library",
  creative: "The Atelier",
  personal: "The Commons",
  health: "The Sanctuary",
  spatial: "The Expanse",
  general: "The Crossroads",
};

export const EVENT_FLAVOR_TEXT: Record<string, { text: string; xp: number }> = {
  "episode.completed": { text: "A new tale inscribed!", xp: 1 },
  "episode.queued": { text: "A rumor reaches the tavern...", xp: 0 },
  "consolidation.phase.merge.completed": { text: "Entities forged together!", xp: 5 },
  "consolidation.phase.dream.completed": { text: "Dream vision: new connections discovered!", xp: 3 },
  "consolidation.phase.prune.completed": { text: "Forgotten memories fade away...", xp: 2 },
  "consolidation.phase.schema.completed": { text: "New ability unlocked!", xp: 10 },
  "consolidation.phase.mature.completed": { text: "Memories have deepened!", xp: 5 },
  "consolidation.completed": { text: "Quest completed! The mind grows stronger.", xp: 15 },
  "consolidation.started": { text: "A new quest begins...", xp: 0 },
  "feedback.recorded": { text: "Preference noted! The path becomes clearer.", xp: 2 },
};

export const QUEST_ACCENT = "#D4A84B";
export const QUEST_GLOW = "rgba(212, 168, 75, 0.15)";

export const ACHIEVEMENT_DEFINITIONS: Array<{
  id: string;
  name: string;
  description: string;
  icon: string;
  condition: (stats: { totalEntities: number; totalRelationships: number; totalEpisodes: number; level: number }) => boolean;
}> = [
  { id: "first_memory", name: "First Memory", description: "Store your first episode", icon: "\u2726",
    condition: (s) => s.totalEpisodes >= 1 },
  { id: "knowledge_seeker", name: "Knowledge Seeker", description: "Reach 50 entities", icon: "\u2605",
    condition: (s) => s.totalEntities >= 50 },
  { id: "web_weaver", name: "Web Weaver", description: "Create 100 relationships", icon: "\u2726",
    condition: (s) => s.totalRelationships >= 100 },
  { id: "memory_palace", name: "Memory Palace", description: "Reach 500 entities", icon: "\u265B",
    condition: (s) => s.totalEntities >= 500 },
  { id: "level_5", name: "Journeyman", description: "Reach level 5", icon: "\u2694",
    condition: (s) => s.level >= 5 },
  { id: "level_10", name: "Veteran", description: "Reach level 10", icon: "\u2655",
    condition: (s) => s.level >= 10 },
  { id: "archivist", name: "Archivist", description: "Record 100 episodes", icon: "\u2721",
    condition: (s) => s.totalEpisodes >= 100 },
  { id: "grand_master", name: "Grand Master", description: "Reach level 20", icon: "\u2654",
    condition: (s) => s.level >= 20 },
];

export const PHASE_DESCRIPTIONS: Record<ConsolidationPhaseName, string> = {
  triage: "Scouting incoming memories for worthy knowledge",
  merge: "Forging duplicate entities into unified beings",
  calibrate: "Attuning to your preferences and desires",
  infer: "Divining hidden connections between entities",
  evidence_adjudication: "Weighing evidence before rewriting the record",
  edge_adjudication: "Judging uncertain links before they become knowledge",
  replay: "Reliving forgotten tales for new insights",
  prune: "Purging dead memories that no longer serve",
  compact: "Condensing access histories into clean records",
  mature: "Evolving young memories into lasting knowledge",
  semanticize: "Transcending episodes to permanent wisdom",
  schema: "Crystallizing recurring patterns into abilities",
  reindex: "Cataloging all changes in the archive",
  graph_embed: "Weaving structural patterns into the fabric",
  microglia: "Cleansing corrupted connections and summaries",
  dream: "Dreaming of new connections across domains",
};
