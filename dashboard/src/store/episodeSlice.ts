import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, EpisodeSlice } from "./types";

export const createEpisodeSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  EpisodeSlice
> = (set, get) => ({
  episodes: [],
  episodeCursor: null,
  hasMoreEpisodes: true,
  isLoadingEpisodes: false,

  loadEpisodes: async (cursor?: string) => {
    if (get().isLoadingEpisodes) return;
    set((s) => {
      s.isLoadingEpisodes = true;
    });
    try {
      const data = await api.getEpisodes({ cursor, limit: 20 });
      set((s) => {
        if (cursor) {
          s.episodes.push(...data.items);
        } else {
          s.episodes = data.items;
        }
        s.episodeCursor = data.nextCursor;
        s.hasMoreEpisodes = data.nextCursor !== null;
        s.isLoadingEpisodes = false;
      });
    } catch {
      set((s) => {
        s.isLoadingEpisodes = false;
      });
    }
  },

  prependEpisode: (episode) =>
    set((s) => {
      s.episodes.unshift(episode);
    }),

  updateEpisodeStatus: (episodeId, status, error) =>
    set((s) => {
      const ep = s.episodes.find((e) => e.episodeId === episodeId);
      if (ep) {
        ep.status = status;
        if (error !== undefined) ep.error = error;
        ep.updatedAt = new Date().toISOString();
      }
    }),
});
