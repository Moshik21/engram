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

export function ChatMessageRenderer({ message }: { message: UIMessage }) {
  if (message.role === "user") {
    const text = getTextContent(message);
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 8 }}>
        <div
          style={{
            maxWidth: "85%",
            padding: "8px 12px",
            borderRadius: "12px 12px 4px 12px",
            background: "rgba(34, 211, 238, 0.1)",
            border: "1px solid rgba(34, 211, 238, 0.15)",
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

  // Assistant message — render parts
  const parts = message.parts ?? [];

  return (
    <div style={{ display: "flex", justifyContent: "flex-start", marginBottom: 8 }}>
      <div
        style={{
          maxWidth: "85%",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        {parts.map((part, i) => {
          if (part.type === "text" && part.text) {
            return (
              <div
                key={i}
                style={{
                  padding: "8px 12px",
                  borderRadius: "12px 12px 12px 4px",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  fontSize: 13,
                  lineHeight: 1.6,
                  color: "var(--text-primary)",
                  wordBreak: "break-word",
                }}
                className="chat-markdown"
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{part.text}</ReactMarkdown>
              </div>
            );
          }
          if (part.type === "dynamic-tool") {
            return <ToolRenderer key={i} part={part} />;
          }
          return null;
        })}
      </div>
    </div>
  );
}
