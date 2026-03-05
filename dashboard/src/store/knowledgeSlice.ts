import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, KnowledgeSlice } from "./types";

export const createKnowledgeSlice: StateCreator<
  EngramStore,
  [["zustand/immer", never]],
  [],
  KnowledgeSlice
> = (set, get) => ({
  knowledgeQuery: "",
  knowledgeResults: [],
  isRecalling: false,
  activeTypeFilter: null,
  entityGroups: {},
  isLoadingEntities: false,
  expandedEntityId: null,
  entityDetail: null,
  inputText: "",
  isSending: false,

  // New state
  pulseEntities: [],
  isPulseLoading: false,
  drawerEntityId: null,
  drawerEntity: null,
  isDrawerLoading: false,
  searchOverlayOpen: false,
  browseOverlayOpen: false,
  intentMode: null,
  confirmDialog: null,
  intentions: [],

  setKnowledgeQuery: (q) => {
    set((s) => {
      s.knowledgeQuery = q;
    });
  },

  executeRecall: async (query) => {
    if (!query.trim()) {
      set((s) => {
        s.knowledgeResults = [];
        s.isRecalling = false;
      });
      return;
    }
    set((s) => {
      s.isRecalling = true;
    });
    try {
      const data = await api.recall({ q: query, limit: 20 });
      set((s) => {
        s.knowledgeResults = data.items;
        s.isRecalling = false;
      });
    } catch {
      set((s) => {
        s.isRecalling = false;
      });
    }
  },

  loadEntityGroups: async () => {
    if (get().isLoadingEntities) return;
    set((s) => {
      s.isLoadingEntities = true;
    });
    try {
      const all = await api.searchEntities({ limit: 100 });
      const groups: Record<string, typeof all> = {};
      for (const entity of all) {
        const type = entity.entityType || "Other";
        if (!groups[type]) groups[type] = [];
        groups[type].push(entity);
      }
      for (const type of Object.keys(groups)) {
        groups[type].sort((a, b) => b.activationScore - a.activationScore);
      }
      set((s) => {
        s.entityGroups = groups;
        s.isLoadingEntities = false;
      });
    } catch {
      set((s) => {
        s.isLoadingEntities = false;
      });
    }
  },

  setActiveTypeFilter: (type) => {
    set((s) => {
      s.activeTypeFilter = type;
    });
  },

  expandEntity: async (id) => {
    if (!id || get().expandedEntityId === id) {
      set((s) => {
        s.expandedEntityId = null;
        s.entityDetail = null;
      });
      return;
    }
    set((s) => {
      s.expandedEntityId = id;
    });
    try {
      const detail = await api.getEntity(id);
      set((s) => {
        s.entityDetail = detail;
      });
    } catch {
      set((s) => {
        s.expandedEntityId = null;
        s.entityDetail = null;
      });
    }
  },

  setInputText: (t) => {
    set((s) => {
      s.inputText = t;
    });
  },

  submitInput: async (text, appendMessages) => {
    const trimmed = text.trim();
    if (!trimmed) return;

    set((s) => {
      s.isSending = true;
      s.inputText = "";
    });

    try {
      // "forget X" or "/forget X" -> confirm dialog
      const forgetMatch = trimmed.match(/^\/?(forget)\s+(.+)/i);
      if (forgetMatch) {
        set((s) => {
          s.intentMode = "forgetting";
          s.confirmDialog = {
            type: "forget",
            entityName: forgetMatch[2],
            title: "Forget Entity",
            message: `Are you sure you want to forget "${forgetMatch[2]}"? This will soft-delete the entity from memory.`,
          };
          s.isSending = false;
          s.intentMode = null;
        });
        return;
      }

      // "remember X" or "/remember X" -> remember
      const rememberMatch = trimmed.match(/^\/?(remember)\s+(.+)/i);
      if (rememberMatch) {
        set((s) => { s.intentMode = "remembering"; });
        await api.remember({ content: rememberMatch[2], source: "dashboard" });
        set((s) => {
          s.intentMode = null;
          s.isSending = false;
        });
        // Add confirmation messages to chat via callback
        if (appendMessages) {
          appendMessages(trimmed, `Remembered: "${rememberMatch[2]}"`);
        }
        return;
      }

      // "/observe X" -> observe
      const observeMatch = trimmed.match(/^\/observe\s+(.+)/i);
      if (observeMatch) {
        set((s) => { s.intentMode = "observing"; });
        await api.observe({ content: observeMatch[1], source: "dashboard" });
        set((s) => {
          s.intentMode = null;
          s.isSending = false;
        });
        if (appendMessages) {
          appendMessages(trimmed, `Observed: "${observeMatch[1]}"`);
        }
        return;
      }
    } catch {
      // silently handle
    } finally {
      set((s) => {
        s.isSending = false;
        s.intentMode = null;
      });
    }
  },

  // --- New actions ---

  loadPulseEntities: async () => {
    set((s) => { s.isPulseLoading = true; });
    try {
      const data = await api.getActivationSnapshot(5);
      set((s) => {
        s.pulseEntities = data.topActivated.map((item) => ({
          entityId: item.entityId,
          name: item.name,
          entityType: item.entityType,
          currentActivation: item.currentActivation,
        }));
        s.isPulseLoading = false;
      });
    } catch {
      set((s) => { s.isPulseLoading = false; });
    }
  },

  setPulseEntities: (entities) => {
    set((s) => {
      s.pulseEntities = entities;
    });
  },

  openDrawer: async (id) => {
    set((s) => {
      s.drawerEntityId = id;
      s.isDrawerLoading = true;
      s.drawerEntity = null;
    });
    try {
      const detail = await api.getEntity(id);
      set((s) => {
        s.drawerEntity = detail;
        s.isDrawerLoading = false;
      });
    } catch {
      set((s) => {
        s.drawerEntityId = null;
        s.drawerEntity = null;
        s.isDrawerLoading = false;
      });
    }
  },

  closeDrawer: () => {
    set((s) => {
      s.drawerEntityId = null;
      s.drawerEntity = null;
      s.isDrawerLoading = false;
    });
  },

  setSearchOverlayOpen: (open) => {
    set((s) => { s.searchOverlayOpen = open; });
  },

  setBrowseOverlayOpen: (open) => {
    set((s) => { s.browseOverlayOpen = open; });
  },

  updateEntity: async (id, patch) => {
    try {
      await api.updateEntity(id, patch);
      // Refresh drawer
      const detail = await api.getEntity(id);
      set((s) => {
        s.drawerEntity = detail;
      });
    } catch {
      // silently handle
    }
  },

  deleteEntity: async (id) => {
    try {
      await api.deleteEntity(id);
      set((s) => {
        s.drawerEntityId = null;
        s.drawerEntity = null;
        s.confirmDialog = null;
      });
      // Refresh entity groups and pulse
      get().loadEntityGroups();
      get().loadPulseEntities();
    } catch {
      // silently handle
    }
  },

  setConfirmDialog: (dialog) => {
    set((s) => { s.confirmDialog = dialog; });
  },

  confirmAction: async () => {
    const dialog = get().confirmDialog;
    if (!dialog) return;

    if (dialog.type === "delete" && dialog.entityId) {
      await get().deleteEntity(dialog.entityId);
    } else if (dialog.type === "forget") {
      try {
        await api.forget({ entity_name: dialog.entityName });
        set((s) => { s.confirmDialog = null; });
        get().loadEntityGroups();
        get().loadPulseEntities();
      } catch {
        set((s) => { s.confirmDialog = null; });
      }
    } else {
      set((s) => { s.confirmDialog = null; });
    }
  },

  handleIntentionEvent: (data: Record<string, unknown>) => {
    const type = data.type as string;
    if (type === "intention.created") {
      // Reload intentions list
      get().loadIntentions();
    } else if (type === "intention.dismissed") {
      set((s) => {
        s.intentions = s.intentions.filter(
          (i) => i.id !== (data.intentionId as string),
        );
      });
    } else if (type === "intention.triggered") {
      set((s) => {
        const idx = s.intentions.findIndex(
          (i) => i.id === (data.intentionId as string),
        );
        if (idx >= 0) {
          s.intentions[idx].fireCount += 1;
        }
      });
    }
  },

  loadIntentions: async () => {
    try {
      const res = await api.getIntentions();
      set((s) => {
        s.intentions = res.intentions ?? [];
      });
    } catch {
      // Non-critical — intentions may not be available yet
    }
  },
});
