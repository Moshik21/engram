import { useState, useRef, useCallback } from "react";
import { useEngramStore } from "../../store";

const SLASH_COMMANDS = [
  { cmd: "/remember", desc: "Store with full extraction" },
  { cmd: "/recall", desc: "Search memories" },
  { cmd: "/forget", desc: "Remove an entity" },
];

export function KnowledgeInputBar() {
  const inputText = useEngramStore((s) => s.inputText);
  const setInputText = useEngramStore((s) => s.setInputText);
  const submitInput = useEngramStore((s) => s.submitInput);
  const isSending = useEngramStore((s) => s.isSending);
  const chatOpen = useEngramStore((s) => s.chatOpen);
  const toggleChat = useEngramStore((s) => s.toggleChat);
  const sendChatMessage = useEngramStore((s) => s.sendChatMessage);
  const isChatStreaming = useEngramStore((s) => s.isChatStreaming);

  const [showCommands, setShowCommands] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleChange = useCallback(
    (value: string) => {
      setInputText(value);
      setShowCommands(value === "/" || (value.startsWith("/") && !value.includes(" ")));
    },
    [setInputText],
  );

  const handleSubmit = useCallback(() => {
    const text = inputText.trim();
    if (!text) return;
    if (chatOpen) {
      sendChatMessage(text);
      setInputText("");
    } else {
      submitInput(text);
    }
    setShowCommands(false);
  }, [inputText, chatOpen, sendChatMessage, setInputText, submitInput]);

  const handleCommandSelect = useCallback(
    (cmd: string) => {
      setInputText(cmd + " ");
      setShowCommands(false);
      inputRef.current?.focus();
    },
    [setInputText],
  );

  const placeholder = chatOpen
    ? "Chat with memory..."
    : "Type to observe... or /remember /recall /forget";

  const isDisabled = isSending || isChatStreaming;

  return (
    <div style={{ position: "relative" }}>
      {/* Slash command palette */}
      {showCommands && !chatOpen && (
        <div
          className="glass-elevated animate-slide-up"
          style={{
            position: "absolute",
            bottom: "100%",
            left: 0,
            right: 60,
            marginBottom: 6,
            borderRadius: "var(--radius-md)",
            overflow: "hidden",
            zIndex: 10,
          }}
        >
          {SLASH_COMMANDS.filter((c) => c.cmd.startsWith(inputText)).map((c) => (
            <button
              key={c.cmd}
              onClick={() => handleCommandSelect(c.cmd)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                width: "100%",
                padding: "8px 14px",
                background: "transparent",
                border: "none",
                borderBottom: "1px solid var(--border-subtle)",
                cursor: "pointer",
                textAlign: "left",
                color: "var(--text-primary)",
                fontSize: 13,
                fontFamily: "var(--font-body)",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "var(--surface-hover)";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
              }}
            >
              <span className="mono" style={{ color: "var(--accent)", fontSize: 12, fontWeight: 500, width: 80 }}>
                {c.cmd}
              </span>
              <span style={{ color: "var(--text-muted)", fontSize: 11 }}>{c.desc}</span>
            </button>
          ))}
        </div>
      )}

      {/* Input bar */}
      <div
        style={{
          display: "flex",
          gap: 8,
          padding: "10px 14px",
          borderTop: "1px solid var(--border)",
          background: "var(--surface-solid)",
        }}
      >
        <input
          ref={inputRef}
          value={inputText}
          onChange={(e) => handleChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !isDisabled) handleSubmit();
            if (e.key === "Escape") setShowCommands(false);
          }}
          placeholder={placeholder}
          disabled={isDisabled}
          style={{
            flex: 1,
            padding: "8px 12px",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text-primary)",
            fontSize: 13,
            fontFamily: "var(--font-body)",
            outline: "none",
            opacity: isDisabled ? 0.6 : 1,
          }}
        />
        <button
          onClick={toggleChat}
          style={{
            padding: "8px 12px",
            background: chatOpen ? "rgba(34, 211, 238, 0.1)" : "transparent",
            border: `1px solid ${chatOpen ? "rgba(34, 211, 238, 0.2)" : "var(--border)"}`,
            borderRadius: "var(--radius-sm)",
            color: chatOpen ? "var(--accent)" : "var(--text-muted)",
            cursor: "pointer",
            fontSize: 13,
            fontFamily: "var(--font-body)",
            transition: "all 0.15s ease",
          }}
        >
          Chat
        </button>
      </div>
    </div>
  );
}
