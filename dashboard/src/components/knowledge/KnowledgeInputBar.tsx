import { useState, useRef, useCallback, useEffect } from "react";
import { useEngramStore } from "../../store";
import { useChatContext } from "./ChatProvider";
import { IntentIndicator } from "./IntentIndicator";

export function KnowledgeInputBar() {
  const submitInput = useEngramStore((s) => s.submitInput);
  const isSending = useEngramStore((s) => s.isSending);
  const setSearchOverlayOpen = useEngramStore((s) => s.setSearchOverlayOpen);
  const setBrowseOverlayOpen = useEngramStore((s) => s.setBrowseOverlayOpen);

  const { sendMessage, setMessages, messages, status } = useChatContext();
  const isStreaming = status === "streaming" || status === "submitted";

  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const appendMessages = useCallback(
    (userText: string, assistantText: string) => {
      setMessages([
        ...messages,
        {
          id: crypto.randomUUID(),
          role: "user" as const,
          parts: [{ type: "text" as const, text: userText }],
        },
        {
          id: crypto.randomUUID(),
          role: "assistant" as const,
          parts: [{ type: "text" as const, text: assistantText }],
        },
      ]);
    },
    [messages, setMessages],
  );

  const onSubmit = useCallback(() => {
    const text = input.trim();
    if (!text) return;

    // Route slash commands through Zustand
    const isSlashCommand = /^\/(remember|observe|forget)\s/i.test(text) ||
      /^(remember|forget)\s/i.test(text);

    if (isSlashCommand) {
      submitInput(text, appendMessages);
      setInput("");
      return;
    }

    // Regular messages go through useChat
    sendMessage({ text });
    setInput("");
  }, [input, submitInput, appendMessages, sendMessage]);

  const isDisabled = isSending || isStreaming;

  // Cmd+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOverlayOpen(true);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [setSearchOverlayOpen]);

  return (
    <div style={{ flexShrink: 0 }}>
      <IntentIndicator />
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
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !isDisabled) onSubmit();
          }}
          placeholder="Ask, remember, or observe..."
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
          onClick={() => setSearchOverlayOpen(true)}
          style={{
            padding: "8px 10px",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text-muted)",
            cursor: "pointer",
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            transition: "all 0.15s ease",
            whiteSpace: "nowrap",
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
          {"\u2318"}K
        </button>
        <button
          onClick={() => setBrowseOverlayOpen(true)}
          style={{
            padding: "8px 12px",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-sm)",
            color: "var(--text-muted)",
            cursor: "pointer",
            fontSize: 12,
            fontFamily: "var(--font-body)",
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
          Browse
        </button>
      </div>
    </div>
  );
}
