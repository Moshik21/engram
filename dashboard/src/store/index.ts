import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { EngramStore } from "./types";
import { createGraphSlice } from "./graphSlice";
import { createSelectionSlice } from "./selectionSlice";
import { createPreferencesSlice } from "./preferencesSlice";
import { createTimeSlice } from "./timeSlice";
import { createEpisodeSlice } from "./episodeSlice";
import { createLifecycleSlice } from "./lifecycleSlice";
import { createEvaluationSlice } from "./evaluationSlice";
import { createStatsSlice } from "./statsSlice";
import { createWsSlice } from "./wsSlice";
import { createActivationSlice } from "./activationSlice";
import { createConsolidationSlice } from "./consolidationSlice";
import { createKnowledgeSlice } from "./knowledgeSlice";
import { createConversationSlice } from "./conversationSlice";
import { createQuestSlice } from "./questSlice";

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
        ...createLifecycleSlice(...a),
        ...createEvaluationSlice(...a),
        ...createWsSlice(...a),
        ...createActivationSlice(...a),
        ...createConsolidationSlice(...a),
        ...createKnowledgeSlice(...a),
        ...createConversationSlice(...a),
        ...createQuestSlice(...a),
      })),
      {
        name: "engram-dashboard",
        version: 5,
        migrate: (persisted: unknown, version: number) => {
          const state = persisted as Record<string, unknown>;
          if (version < 3) {
            state.graphMaxNodes = 1500;
          }
          if (version < 4) {
            state.lastAtlasVisitAt = null;
            state.lastAtlasSnapshotId = null;
          }
          if (version < 5) {
            state.dashboardMode = "observatory";
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
          lastAtlasVisitAt: s.lastAtlasVisitAt,
          lastAtlasSnapshotId: s.lastAtlasSnapshotId,
          dashboardMode: s.dashboardMode,
        }),
      },
    ),
  ),
);
