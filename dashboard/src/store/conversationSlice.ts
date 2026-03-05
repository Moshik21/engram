import type { StateCreator } from "zustand";
import type { EngramStore, ConversationSlice } from "./types";
import { api } from "../api/client";

export const createConversationSlice: StateCreator<
  EngramStore,
  [["zustand/devtools", never], ["zustand/persist", unknown], ["zustand/immer", never]],
  [],
  ConversationSlice
> = (set) => ({
  conversations: [],
  isLoadingConversations: false,
  activeConversationId: null,
  conversationSidebarOpen: false,

  loadConversations: async () => {
    set((s) => { s.isLoadingConversations = true; });
    try {
      const data = await api.listConversations(50);
      set((s) => {
        s.conversations = data.conversations;
        s.isLoadingConversations = false;
      });
    } catch {
      set((s) => { s.isLoadingConversations = false; });
    }
  },

  setActiveConversation: (id) => {
    set((s) => { s.activeConversationId = id; });
  },

  setConversationId: (id) => {
    set((s) => { s.activeConversationId = id; });
  },

  toggleConversationSidebar: () => {
    set((s) => { s.conversationSidebarOpen = !s.conversationSidebarOpen; });
  },

  startNewConversation: () => {
    set((s) => { s.activeConversationId = null; });
  },
});
