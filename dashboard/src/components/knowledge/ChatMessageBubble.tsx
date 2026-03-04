import { InlineEntityCard, type InlineEntitySource } from "./InlineEntityCard";

interface LegacyChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: InlineEntitySource[];
  timestamp: number;
}

export function ChatMessageBubble({ msg }: { msg: LegacyChatMessage }) {
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
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 8 }}>
            {msg.sources.map((src, i) => (
              <InlineEntityCard key={i} source={src} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
