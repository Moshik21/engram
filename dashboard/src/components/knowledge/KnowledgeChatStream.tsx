import { useRef, useEffect } from "react";
import { useChatContext } from "./ChatProvider";
import { ChatMessageRenderer } from "./ChatMessageRenderer";

export function KnowledgeChatStream() {
  const { messages, status, setMessages } = useChatContext();
  const scrollRef = useRef<HTMLDivElement>(null);

  const isStreaming = status === "streaming" || status === "submitted";

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const clearChat = () => setMessages([]);

  return (
    <div
      ref={scrollRef}
      style={{
        flex: 1,
        overflowY: "auto",
        position: "relative",
      }}
    >
      {/* Clear button */}
      {messages.length > 0 && (
        <button
          onClick={clearChat}
          style={{
            position: "sticky",
            top: 8,
            float: "right",
            marginRight: 16,
            background: "none",
            border: "none",
            color: "var(--text-muted)",
            cursor: "pointer",
            fontSize: 10,
            fontFamily: "var(--font-mono)",
            zIndex: 2,
            padding: "4px 8px",
            borderRadius: "var(--radius-xs)",
            transition: "color 0.15s",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = "var(--text-secondary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = "var(--text-muted)";
          }}
        >
          Clear
        </button>
      )}

      <div
        style={{
          maxWidth: 720,
          margin: "0 auto",
          padding: "20px 20px 16px",
        }}
      >
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <>
            {messages.map((msg) => (
              <ChatMessageRenderer key={msg.id} message={msg} />
            ))}
            {/* Streaming indicator */}
            {isStreaming && (
              <div style={{ display: "flex", justifyContent: "flex-start", marginTop: 4, marginBottom: 8 }}>
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "var(--accent)",
                    boxShadow: "0 0 8px var(--accent-glow-strong)",
                    animation: "pulse-soft 1.5s ease-in-out infinite",
                  }}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div
      className="animate-fade-in"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        paddingTop: "15vh",
        textAlign: "center",
      }}
    >
      {/* Neural ring animation */}
      <div style={{ position: "relative", width: 80, height: 80, marginBottom: 24 }}>
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              position: "absolute",
              inset: i * 8,
              borderRadius: "50%",
              border: "1px solid var(--accent)",
              opacity: 0.15 + i * 0.1,
              animation: `glow-ring ${3 + i * 0.5}s ease-in-out infinite`,
              animationDelay: `${i * 0.4}s`,
            }}
          />
        ))}
        <div
          style={{
            position: "absolute",
            inset: 28,
            borderRadius: "50%",
            background: "var(--accent-glow)",
            boxShadow: "0 0 20px var(--accent-glow)",
          }}
        />
      </div>

      <h2
        className="display"
        style={{
          fontSize: 22,
          fontWeight: 400,
          color: "var(--text-primary)",
          margin: 0,
          marginBottom: 8,
        }}
      >
        Ask your memory
      </h2>
      <p
        style={{
          fontSize: 12,
          color: "var(--text-muted)",
          maxWidth: 320,
          lineHeight: 1.6,
        }}
      >
        Ask questions, remember facts, or browse your knowledge graph.
        Everything you store builds a living memory.
      </p>
    </div>
  );
}
