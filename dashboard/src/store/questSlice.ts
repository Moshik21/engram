import type { StateCreator } from "zustand";
import type { EngramStore, GraphStats, PlayerClass, QuestSlice } from "./types";

function classifyDomain(stats: GraphStats): PlayerClass {
  const types = stats.entityTypeCounts || {};
  const domains: Record<string, number> = {
    technical: (types["Technology"] ?? 0) + (types["Software"] ?? 0) + (types["Project"] ?? 0),
    knowledge: types["Concept"] ?? 0,
    creative: (types["CreativeWork"] ?? 0) + (types["Article"] ?? 0),
    personal: (types["Person"] ?? 0) + (types["Event"] ?? 0) + (types["Goal"] ?? 0),
    health: (types["HealthCondition"] ?? 0) + (types["BodyPart"] ?? 0),
    spatial: (types["Organization"] ?? 0) + (types["Location"] ?? 0),
  };
  const sorted = Object.entries(domains).sort((a, b) => b[1] - a[1]);
  if (!sorted.length || sorted[0][1] === 0) return "Polymath";
  const top = sorted[0][0];
  const classMap: Record<string, PlayerClass> = {
    technical: "Artificer",
    knowledge: "Sage",
    creative: "Bard",
    personal: "Diplomat",
    health: "Alchemist",
    spatial: "Cartographer",
  };
  // Check if balanced (top domain < 40% of total)
  const total = sorted.reduce((s, [, v]) => s + v, 0);
  if (total > 0 && sorted[0][1] / total < 0.4) return "Polymath";
  return classMap[top] ?? "Polymath";
}

export const createQuestSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  QuestSlice
> = (set) => ({
  dashboardMode: "observatory",
  playerStats: {
    level: 1,
    xp: 0,
    xpToNext: 50,
    hp: 100,
    mp: 50,
    gold: 0,
    playerClass: "Polymath",
    domainScores: {},
  },
  questEvents: [],
  feedbackPositive: 0,
  feedbackNegative: 0,

  setDashboardMode: (mode) =>
    set((s) => {
      s.dashboardMode = mode;
    }),

  computePlayerStats: (stats) =>
    set((s) => {
      const totalEntities = stats.totalEntities ?? 0;
      const level = Math.floor(totalEntities / 50) + 1;
      const xp = totalEntities % 50;

      // HP: graph health (connectivity ratio)
      const totalEdges = stats.totalRelationships ?? 0;
      const hp = totalEntities > 0 ? Math.min(100, Math.round((totalEdges / totalEntities) * 50)) : 100;

      // Gold: total episodes (successful recalls)
      const gold = stats.totalEpisodes ?? 0;

      // Domain scores from entity type counts
      const types = stats.entityTypeCounts || {};
      const domainScores: Record<string, number> = {
        technical: (types["Technology"] ?? 0) + (types["Software"] ?? 0) + (types["Project"] ?? 0),
        knowledge: types["Concept"] ?? 0,
        creative: (types["CreativeWork"] ?? 0) + (types["Article"] ?? 0),
        personal: (types["Person"] ?? 0) + (types["Event"] ?? 0) + (types["Goal"] ?? 0),
        health: (types["HealthCondition"] ?? 0) + (types["BodyPart"] ?? 0),
        spatial: (types["Organization"] ?? 0) + (types["Location"] ?? 0),
      };

      // MP: morale from net positive feedback (0-100)
      const netSentiment = s.feedbackPositive - s.feedbackNegative;
      const total = s.feedbackPositive + s.feedbackNegative;
      const mp = total > 0 ? Math.round(Math.max(0, Math.min(100, 50 + (netSentiment / total) * 50))) : 50;

      s.playerStats = {
        level,
        xp,
        xpToNext: 50,
        hp,
        mp,
        gold,
        playerClass: classifyDomain(stats),
        domainScores,
      };
    }),

  addQuestEvent: (text, xp) =>
    set((s) => {
      s.questEvents.unshift({
        id: `qe_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        text,
        xp,
        timestamp: Date.now(),
      });
      // Keep only last 50 events
      if (s.questEvents.length > 50) s.questEvents.length = 50;
    }),

  recordFeedback: (positive) =>
    set((s) => {
      if (positive) {
        s.feedbackPositive += 1;
      } else {
        s.feedbackNegative += 1;
      }
    }),
});
