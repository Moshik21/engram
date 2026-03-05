import { useState } from "react";
import { useEngramStore } from "../../store";
import { useChatContext } from "./ChatProvider";
import type { SearchResult } from "../../store/types";

type GroupMode = "date" | "topic";

type Conversation = {
  id: string;
  title: string | null;
  sessionDate: string;
  updatedAt: string;
  entityIds: string[];
};

function groupByDate(conversations: Conversation[]): Record<string, Conversation[]> {
  const today = new Date().toISOString().slice(0, 10);
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10);
  const groups: Record<string, Conversation[]> = {};

  for (const conv of conversations) {
    let label: string;
    if (conv.sessionDate === today) label = "Today";
    else if (conv.sessionDate === yesterday) label = "Yesterday";
    else label = conv.sessionDate;
    if (!groups[label]) groups[label] = [];
    groups[label].push(conv);
  }
  return groups;
}

function groupByTopic(
  conversations: Conversation[],
  entityGroups: Record<string, SearchResult[]>,
): Record<string, Conversation[]> {
  // Build entity ID -> name map from all entity groups
  const idToName: Record<string, string> = {};
  for (const entities of Object.values(entityGroups)) {
    for (const e of entities) {
      idToName[e.id] = e.name;
    }
  }

  const groups: Record<string, Conversation[]> = {};
  const ungrouped: Conversation[] = [];

  for (const conv of conversations) {
    if (conv.entityIds.length === 0) {
      ungrouped.push(conv);
      continue;
    }
    // Place under the first entity name we can resolve
    let placed = false;
    for (const eid of conv.entityIds) {
      const name = idToName[eid];
      if (name) {
        if (!groups[name]) groups[name] = [];
        groups[name].push(conv);
        placed = true;
        break;
      }
    }
    if (!placed) ungrouped.push(conv);
  }

  // Fall back: group ungrouped by date so it's not all "Other"
  if (ungrouped.length > 0) {
    const dateGroups = groupByDate(ungrouped);
    for (const [label, convs] of Object.entries(dateGroups)) {
      const key = Object.keys(groups).length === 0 ? label : `${label}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push(...convs);
    }
  }

  return groups;
}

export function ConversationSidebar() {
  const conversations = useEngramStore((s) => s.conversations);
  const activeConversationId = useEngramStore((s) => s.activeConversationId);
  const setActiveConversation = useEngramStore((s) => s.setActiveConversation);
  const startNewConversation = useEngramStore((s) => s.startNewConversation);
  const entityGroups = useEngramStore((s) => s.entityGroups);

  const { setMessages } = useChatContext();

  const [mode, setMode] = useState<GroupMode>("date");

  const grouped =
    mode === "date"
      ? groupByDate(conversations)
      : groupByTopic(conversations, entityGroups);

  const handleSelect = (id: string) => {
    setActiveConversation(id);
  };

  const handleNew = () => {
    startNewConversation();
    setMessages([]);
  };

  // Count how many conversations have entity tags (for topic mode indicator)
  const taggedCount = conversations.filter((c) => c.entityIds.length > 0).length;

  return (
    <div
      style={{
        width: 240,
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        background: "rgba(0,0,0,0.15)",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "10px 12px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          gap: 6,
        }}
      >
        <button
          onClick={() => setMode(mode === "date" ? "topic" : "date")}
          style={{
            flex: 1,
            padding: "4px 8px",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-xs)",
            color: "var(--text-muted)",
            cursor: "pointer",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            textAlign: "left",
            transition: "all 0.15s ease",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--border-hover)";
            e.currentTarget.style.color = "var(--text-secondary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "var(--border)";
            e.currentTarget.style.color = "var(--text-muted)";
          }}
        >
          {mode === "date" ? "By Date" : `By Topic (${taggedCount})`}
        </button>
        <button
          onClick={handleNew}
          style={{
            padding: "4px 10px",
            background: "var(--accent-glow)",
            border: "none",
            borderRadius: "var(--radius-xs)",
            color: "var(--text-primary)",
            cursor: "pointer",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            fontWeight: 600,
          }}
        >
          New
        </button>
      </div>

      {/* Conversation list */}
      <div style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
        {Object.entries(grouped).map(([groupLabel, convs]) => (
          <div key={groupLabel}>
            <div
              style={{
                padding: "6px 12px 2px",
                fontSize: 9,
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
                textTransform: "uppercase",
                letterSpacing: "0.05em",
              }}
            >
              {groupLabel}
            </div>
            {convs.map((conv) => (
              <button
                key={conv.id}
                onClick={() => handleSelect(conv.id)}
                style={{
                  display: "block",
                  width: "100%",
                  padding: "6px 12px",
                  background:
                    conv.id === activeConversationId
                      ? "rgba(255,255,255,0.06)"
                      : "transparent",
                  border: "none",
                  borderLeft:
                    conv.id === activeConversationId
                      ? "2px solid var(--accent)"
                      : "2px solid transparent",
                  color: "var(--text-secondary)",
                  cursor: "pointer",
                  fontSize: 11,
                  fontFamily: "var(--font-body)",
                  textAlign: "left",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => {
                  if (conv.id !== activeConversationId)
                    e.currentTarget.style.background = "rgba(255,255,255,0.03)";
                }}
                onMouseLeave={(e) => {
                  if (conv.id !== activeConversationId)
                    e.currentTarget.style.background = "transparent";
                }}
              >
                {conv.title || "Untitled"}
              </button>
            ))}
          </div>
        ))}
        {conversations.length === 0 && (
          <div
            style={{
              padding: "20px 12px",
              fontSize: 11,
              color: "var(--text-muted)",
              textAlign: "center",
            }}
          >
            No conversations yet
          </div>
        )}
      </div>
    </div>
  );
}
