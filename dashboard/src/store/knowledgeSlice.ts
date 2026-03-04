import type { StateCreator } from "zustand";
import { api } from "../api/client";
import type { EngramStore, KnowledgeSlice, ChatMessage } from "./types";

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
  chatMessages: [],
  isChatStreaming: false,
  chatOpen: false,

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
      // Sort each group by activation descending
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

  submitInput: async (text) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    set((s) => {
      s.isSending = true;
      s.inputText = "";
    });
    try {
      if (trimmed.startsWith("/remember ")) {
        await api.remember({ content: trimmed.slice(10), source: "dashboard" });
      } else if (trimmed.startsWith("/recall ")) {
        const query = trimmed.slice(8);
        set((s) => {
          s.knowledgeQuery = query;
        });
        await get().executeRecall(query);
      } else if (trimmed.startsWith("/forget ")) {
        await api.forget({ entity_name: trimmed.slice(8) });
        get().loadEntityGroups();
      } else {
        await api.observe({ content: trimmed, source: "dashboard" });
      }
    } catch {
      // silently handle
    } finally {
      set((s) => {
        s.isSending = false;
      });
    }
  },

  sendChatMessage: async (message) => {
    const trimmed = message.trim();
    if (!trimmed || get().isChatStreaming) return;

    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
      timestamp: Date.now(),
    };

    const assistantId = crypto.randomUUID();

    set((s) => {
      s.chatMessages.push(userMsg);
      s.chatMessages.push({
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
      });
      s.isChatStreaming = true;
      s.inputText = "";
    });

    try {
      const history = get()
        .chatMessages.filter((m) => m.id !== assistantId && m.role !== "assistant" || m.content)
        .slice(-10)
        .map((m) => ({ role: m.role, content: m.content }));

      const response = await fetch("/api/knowledge/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, history }),
      });

      if (!response.ok) throw new Error(`Chat error: ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") continue;

          try {
            const event = JSON.parse(payload);
            if (event.type === "text") {
              set((s) => {
                const msg = s.chatMessages.find((m) => m.id === assistantId);
                if (msg) msg.content += event.content;
              });
            } else if (event.type === "sources") {
              set((s) => {
                const msg = s.chatMessages.find((m) => m.id === assistantId);
                if (msg) {
                  msg.sources = event.items.map(
                    (item: { name?: string; content?: string; score?: number }) => ({
                      name: item.name || (item.content ? item.content.slice(0, 40) + "..." : "Memory"),
                      score: item.score,
                    }),
                  );
                }
              });
            } else if (event.type === "error") {
              set((s) => {
                const msg = s.chatMessages.find((m) => m.id === assistantId);
                if (msg) msg.content = `Error: ${event.content}`;
              });
            }
          } catch {
            // skip malformed SSE events
          }
        }
      }
    } catch {
      set((s) => {
        const msg = s.chatMessages.find((m) => m.id === assistantId);
        if (msg && !msg.content) msg.content = "Failed to get response.";
      });
    } finally {
      set((s) => {
        s.isChatStreaming = false;
      });
    }
  },

  toggleChat: () => {
    set((s) => {
      s.chatOpen = !s.chatOpen;
    });
  },

  clearChat: () => {
    set((s) => {
      s.chatMessages = [];
    });
  },
});
