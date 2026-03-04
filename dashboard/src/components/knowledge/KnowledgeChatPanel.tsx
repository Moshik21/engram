import { useRef, useEffect } from "react";
import { useEngramStore } from "../../store";
import { ChatMessageBubble } from "./ChatMessageBubble";

export function KnowledgeChatPanel() {
  const chatOpen = useEngramStore((s) => s.chatOpen);
  const messages = useEngramStore((s) => s.chatMessages);
  const isChatStreaming = useEngramStore((s) => s.isChatStreaming);
  const clearChat = useEngramStore((s) => s.clearChat);
  const toggleChat = useEngramStore((s) => s.toggleChat);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  if (!chatOpen) return null;

  return (
    <div
      className="glass-elevated animate-slide-right"
      style={{
        width: 320,
        flexShrink: 0,
        borderRadius: "var(--radius-lg)",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: "12px 14px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: isChatStreaming ? "var(--accent)" : "var(--accent-dim)",
              boxShadow: isChatStreaming ? "0 0 8px var(--accent-glow-strong)" : "none",
              animation: isChatStreaming ? "glow-ring 2s ease-in-out infinite" : "none",
            }}
          />
          <span className="label" style={{ fontSize: 10 }}>Memory Chat</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {messages.length > 0 && (
            <button
              onClick={clearChat}
              style={{
                background: "none",
                border: "none",
                color: "var(--text-muted)",
                cursor: "pointer",
                fontSize: 10,
                fontFamily: "var(--font-mono)",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.color = "var(--text-secondary)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.color = "var(--text-muted)"; }}
            >
              Clear
            </button>
          )}
          <button
            onClick={toggleChat}
            style={{
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: 14,
              padding: "0 2px",
              lineHeight: 1,
            }}
          >
            &times;
          </button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "12px 10px",
        }}
      >
        {messages.length === 0 && (
          <div style={{ textAlign: "center", padding: "40px 16px" }}>
            <p style={{ color: "var(--text-muted)", fontSize: 12 }}>
              Ask questions about stored memories
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessageBubble key={msg.id} msg={msg} />
        ))}
      </div>
    </div>
  );
}
