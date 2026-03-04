import type { ChatMessage } from "../../store/types";
import { entityColor } from "../../lib/colors";

export function ChatMessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: 8,
      }}
    >
      <div
        style={{
          maxWidth: "85%",
          padding: "8px 12px",
          borderRadius: isUser ? "12px 12px 4px 12px" : "12px 12px 12px 4px",
          background: isUser ? "rgba(34, 211, 238, 0.1)" : "var(--surface)",
          border: `1px solid ${isUser ? "rgba(34, 211, 238, 0.15)" : "var(--border)"}`,
          fontSize: 13,
          lineHeight: 1.5,
          color: "var(--text-primary)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {msg.content}

        {msg.sources && msg.sources.length > 0 && (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 6 }}>
            {msg.sources.map((src, i) => (
              <span
                key={i}
                className="mono"
                style={{
                  fontSize: 9,
                  padding: "2px 6px",
                  borderRadius: 99,
                  background: `${entityColor(src.entityType ?? "Other")}15`,
                  color: entityColor(src.entityType ?? "Other"),
                  border: `1px solid ${entityColor(src.entityType ?? "Other")}20`,
                }}
              >
                {src.name}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
