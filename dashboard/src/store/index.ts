import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { EngramStore } from "./types";
import { createGraphSlice } from "./graphSlice";
import { createSelectionSlice } from "./selectionSlice";
import { createPreferencesSlice } from "./preferencesSlice";
import { createTimeSlice } from "./timeSlice";
import { createEpisodeSlice } from "./episodeSlice";
import { createStatsSlice } from "./statsSlice";
import { createWsSlice } from "./wsSlice";
import { createActivationSlice } from "./activationSlice";
import { createConsolidationSlice } from "./consolidationSlice";
import { createKnowledgeSlice } from "./knowledgeSlice";
import { createConversationSlice } from "./conversationSlice";

export const useEngramStore = create<EngramStore>()(
  devtools(
    persist(
      immer((...a) => ({
        ...createGraphSlice(...a),
        ...createSelectionSlice(...a),
        ...createPreferencesSlice(...a),
        ...createTimeSlice(...a),
        ...createEpisodeSlice(...a),
        ...createStatsSlice(...a),
        ...createWsSlice(...a),
        ...createActivationSlice(...a),
        ...createConsolidationSlice(...a),
        ...createKnowledgeSlice(...a),
        ...createConversationSlice(...a),
      })),
      {
        name: "engram-dashboard",
        version: 2,
        migrate: (persisted: unknown, version: number) => {
          const state = persisted as Record<string, unknown>;
          if (version < 2) {
            // Raise graphMaxNodes — LOD now handles visibility budgeting
            state.graphMaxNodes = 50000;
          }
          return state;
        },
        partialize: (s) => ({
          currentView: s.currentView,
          renderMode: s.renderMode,
          showActivationHeatmap: s.showActivationHeatmap,
          showEdgeLabels: s.showEdgeLabels,
          darkMode: s.darkMode,
          graphMaxNodes: s.graphMaxNodes,
        }),
      },
    ),
  ),
);
