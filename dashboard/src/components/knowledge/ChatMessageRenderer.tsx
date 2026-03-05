import { memo } from "react";
import type { UIMessage } from "ai";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolRenderer } from "./tools/ToolRenderer";

/** Extract plain text from a UIMessage's parts. */
function getTextContent(message: UIMessage): string {
  return message.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

const TOOL_LABELS: Record<string, string> = {
  show_entities: "Relevant Entities",
  show_relationship_graph: "Relationship Graph",
  show_facts: "Facts",
  show_activation_chart: "Activation Levels",
  show_timeline: "Memory Timeline",
};

export const ChatMessageRenderer = memo(function ChatMessageRenderer({ message }: { message: UIMessage }) {
  if (message.role === "user") {
    const text = getTextContent(message);
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 10 }}>
        <div
          style={{
            maxWidth: "85%",
            padding: "8px 14px",
            borderRadius: "12px 12px 4px 12px",
            background: "rgba(34, 211, 238, 0.08)",
            border: "1px solid rgba(34, 211, 238, 0.12)",
            fontSize: 13,
            lineHeight: 1.5,
            color: "var(--text-primary)",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {text}
        </div>
      </div>
    );
  }

  // Assistant message — separate text parts from tool parts
  const parts = message.parts ?? [];
  const hasTools = parts.some((p) => p.type === "dynamic-tool");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8, marginBottom: 10 }}>
      {parts.map((part, i) => {
        if (part.type === "text" && part.text) {
          return (
            <div
              key={i}
              style={{ display: "flex", justifyContent: "flex-start" }}
            >
              <div
                style={{
                  maxWidth: hasTools ? "100%" : "85%",
                  padding: "10px 14px",
                  borderRadius: "12px 12px 12px 4px",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  wordBreak: "break-word",
                }}
                className="chat-markdown"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.text}</ReactMarkdown>
              </div>
            </div>
          );
        }
        if (part.type === "dynamic-tool") {
          const label = TOOL_LABELS[part.toolName] || part.toolName;
          return (
            <div
              key={part.toolCallId}
              style={{
                width: "100%",
                animation: "slide-in-up 0.25s ease-out both",
              }}
            >
              {/* Tool header label */}
              <div
                style={{
                  fontSize: 9,
                  fontFamily: "var(--font-mono)",
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--text-muted)",
                  marginBottom: 4,
                  paddingLeft: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                }}
              >
                <span
                  style={{
                    width: 4,
                    height: 4,
                    borderRadius: "50%",
                    background: "var(--accent)",
                    opacity: 0.5,
                    flexShrink: 0,
                  }}
                />
                {label}
              </div>
              <ToolRenderer part={part} />
            </div>
          );
        }
        return null;
      })}
    </div>
  );
});
