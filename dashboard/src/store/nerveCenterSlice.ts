import type { StateCreator } from "zustand";
import type { EngramStore, GraphStats, NeuralSpecialization, NerveCenterSlice } from "./types";

function classifyDomain(stats: GraphStats): NeuralSpecialization {
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
  const specializationMap: Record<string, NeuralSpecialization> = {
    technical: "Architect",
    knowledge: "Synthesizer",
    creative: "Narrator",
    personal: "Integrator",
    health: "Biochemist",
    spatial: "Topologist",
  };
  // Check if balanced (top domain < 40% of total)
  const total = sorted.reduce((s, [, v]) => s + v, 0);
  if (total > 0 && sorted[0][1] / total < 0.4) return "Polymath";
  return specializationMap[top] ?? "Polymath";
}

export const createNerveCenterSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  NerveCenterSlice
> = (set) => ({
  dashboardMode: "observatory",
  cerebralStats: {
    level: 1,
    plasticity: 0,
    plasticityToNext: 50,
    homeostasis: 100,
    morale: 50,
    synapticCredits: 0,
    specialization: "Polymath",
    domainScores: {},
  },
  synapticEvents: [],
  feedbackPositive: 0,
  feedbackNegative: 0,

  setDashboardMode: (mode) =>
    set((s) => {
      s.dashboardMode = mode;
    }),

  computeCerebralStats: (stats) =>
    set((s) => {
      const totalEntities = stats.totalEntities ?? 0;
      const level = Math.floor(totalEntities / 50) + 1;
      const plasticity = totalEntities % 50;

      // Homeostasis: graph health (connectivity ratio)
      const totalEdges = stats.totalRelationships ?? 0;
      const homeostasis = totalEntities > 0 ? Math.min(100, Math.round((totalEdges / totalEntities) * 50)) : 100;

      // Synaptic Credits: total episodes (successful recalls)
      const synapticCredits = stats.totalEpisodes ?? 0;

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

      // Morale: from net positive feedback (0-100)
      const netSentiment = s.feedbackPositive - s.feedbackNegative;
      const total = s.feedbackPositive + s.feedbackNegative;
      const morale = total > 0 ? Math.round(Math.max(0, Math.min(100, 50 + (netSentiment / total) * 50))) : 50;

      s.cerebralStats = {
        level,
        plasticity,
        plasticityToNext: 50,
        homeostasis,
        morale,
        synapticCredits,
        specialization: classifyDomain(stats),
        domainScores,
      };
    }),

  addSynapticEvent: (text, plasticity) =>
    set((s) => {
      s.synapticEvents.unshift({
        id: `se_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        text,
        plasticity,
        timestamp: Date.now(),
      });
      // Keep only last 50 events
      if (s.synapticEvents.length > 50) s.synapticEvents.length = 50;
    }),

  recordFeedback: (positive) =>
    set((s) => {
      if (positive) {
        s.feedbackPositive += 1;
      } else {
        s.feedbackNegative += 1;
      }
    }),

  notifications: [],
  loadNotifications: async () => {
    try {
      const { api } = (await import("../api/client"));
      const { notifications } = await api.getNotifications();
      set((s) => {
        s.notifications = notifications;
      });
    } catch (err) {
      console.error("Failed to load notifications", err);
    }
  },
  dismissNotifications: async (ids) => {
    try {
      const { api } = (await import("../api/client"));
      await api.dismissNotifications(ids);
      set((s) => {
        s.notifications = s.notifications.filter((n) => !ids.includes(n.id));
      });
    } catch (err) {
      console.error("Failed to dismiss notifications", err);
    }
  },

  adjudicationRequests: [],
  isAdjudicating: false,
  loadAdjudications: async () => {
    try {
      const { api } = (await import("../api/client"));
      const { requests } = await api.getAdjudications();
      set((s) => {
        s.adjudicationRequests = requests;
      });
    } catch (err) {
      console.error("Failed to load adjudications", err);
    }
  },
  resolveAdjudication: async (body) => {
    set((s) => {
      s.isAdjudicating = true;
    });
    try {
      const { api } = (await import("../api/client"));
      await api.adjudicate(body);
      set((s) => {
        s.adjudicationRequests = s.adjudicationRequests.filter(
          (r) => r.request_id !== body.request_id
        );
        s.isAdjudicating = false;
      });
    } catch (err) {
      console.error("Failed to resolve adjudication", err);
      set((s) => {
        s.isAdjudicating = false;
      });
    }
  },

  selectedNeuralLayer: "activity",
  setSelectedNeuralLayer: (layer) =>
    set((s) => {
      s.selectedNeuralLayer = layer;
    }),
});
